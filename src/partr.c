// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "julia.h"
#include "julia_internal.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include "synctreepool.h"
#include "taskpools.h"


/* the `start` task */
ptask_t *start_task;

/* sticky task queues need to be visible to all threads */
ptask_t  ***all_taskqs;
int8_t   **all_taskq_locks;

/* forward declare thread function */
void *partr_thread(void *arg_);

/* internally used to indicate a yield occurred in the runtime itself */
static const int64_t yield_from_sync = 1;

/* initialization thread barrier */
static int volatile barcnt;
static int volatile barsense = 1;

#define BARRIER_INIT()          barcnt=nthreads
#define BARRIER_THREAD_DECL     int mysense = 1
#define BARRIER() do {                                                  \
    mysense = !mysense;                                                 \
    if (!__atomic_sub_fetch(&barcnt, 1, __ATOMIC_SEQ_CST)) {            \
        barcnt = nthreads;                                              \
        barsense = mysense;                                             \
    } else while (barsense != mysense);                                 \
} while(0)


/* thread function argument */
typedef struct lthread_arg_tag {
    int16_t             tid;
    int8_t              exclusive;
    hwloc_topology_t    topology;
    hwloc_cpuset_t      cpuset;
} lthread_arg_t;


/*  partr_init() -- initialization entry point
 */
void partr_init()
{
    char *cp;

    /* get requested # threads */
    nthreads = DEFAULT_NUM_THREADS;
    cp = getenv(NUM_THREADS_NAME);
    if (cp)
        nthreads = strtol(cp, NULL, 10);

    /* check if we have exclusive use of the machine */
    int exclusive = DEFAULT_MACHINE_EXCLUSIVE;
    cp = getenv(MACHINE_EXCLUSIVE_NAME);
    if (cp)
        exclusive = strtol(cp, NULL, 10);

    /* check machine topology */
    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);
    int core_depth = hwloc_get_type_or_below_depth(topology, HWLOC_OBJ_CORE);
    unsigned ncores = hwloc_get_nbobjs_by_depth(topology, core_depth);
    int pu_depth = hwloc_get_type_or_below_depth(topology, HWLOC_OBJ_PU);
    unsigned npus = hwloc_get_nbobjs_by_depth(topology, pu_depth);

    /* some sanity checks */
    if (nthreads > npus) {
        nthreads = npus;
    }
    if (nthreads < 1) {
        nthreads = ncores;
    }
    int depth;
    if (nthreads <= ncores) {
        depth = core_depth;
    }
    else {
        depth = pu_depth;
    }

    /* set affinity if we have exclusive use of the machine */
    hwloc_obj_t obj;
    hwloc_cpuset_t cpuset;
    if (exclusive) {
        /* rebind this thread to the first core/PU */
        obj = hwloc_get_obj_by_depth(topology, depth, 0);
        assert(obj != NULL);
        cpuset = hwloc_bitmap_dup(obj->cpuset);
        /* hwloc_bitmap_singlify(cpuset); */
        hwloc_set_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD);
        hwloc_bitmap_free(cpuset);
    }

    tid = 0;
    seed_cong(&rngseed);

    /* initialize task pools */
    taskpools_init();

    /* initialize sync trees */
    synctreepool_init();

    /* initialize task multiqueue */
    multiq_init();

    /* initialize libconcurrent */
    concurrent_init();

    /* allocate per-thread task queues, for sticky tasks */
    all_taskqs = (ptask_t ***)_mm_malloc(nthreads * sizeof(ptask_t **), 64);
    all_taskq_locks = (int8_t **)_mm_malloc(nthreads * sizeof(int8_t *), 64);

    /* start threads */
    BARRIER_THREAD_DECL;
    BARRIER_INIT();

    pthread_t pthread_id;
    pthread_attr_t pthread_attr;

    pthread_attr_init(&pthread_attr);
    pthread_attr_setdetachstate(&pthread_attr, PTHREAD_CREATE_DETACHED);

    for (int16_t i = 1;  i < nthreads;  ++i) {
        lthread_arg_t *targ = (lthread_arg_t *)calloc(1, sizeof(lthread_arg_t));
        targ->tid = i;
        targ->exclusive = exclusive;

        if (exclusive) {
            /* tell the thread which core to bind to */
            obj = hwloc_get_obj_by_depth(topology, depth, i);
            cpuset = hwloc_bitmap_dup(obj->cpuset);
            targ->topology = topology;
            targ->cpuset = cpuset;
        }
        pthread_create(&pthread_id, &pthread_attr, partr_thread, targ);
    }
    pthread_attr_destroy(&pthread_attr);

    /* allocate this thread's sticky task queue pointer and initialize the lock */
    taskq_lock = (int8_t *)_mm_malloc(sizeof(int8_t) + sizeof(ptask_t *), 64);
    taskq = (ptask_t **)(taskq_lock + sizeof(int8_t));
    __atomic_clear(taskq_lock, __ATOMIC_RELAXED);
    *taskq = NULL;
    all_taskqs[tid] = taskq;
    all_taskq_locks[tid] = taskq_lock;

    /* wait for all threads to start up and bind to their CPUs */
    BARRIER();
    hwloc_topology_destroy(topology);
}


/*  partr_shutdown() -- shutdown all threads and clean up
 */
void partr_shutdown()
{
    /* create and add 'nthreads' terminate tasks */
    for (int64_t i = 0;  i < nthreads;  ++i) {
        ptask_t *task = task_alloc();
        if (task == NULL)
            break;
        task->settings = TASK_TERMINATE;
        if (multiq_insert(task, tid) != 0) {
            task_free(task);
            break;
        }
    }

    /* wait for all threads to shut down */
    sleep(1);

    /* free task queues and their locks */
    _mm_free(taskq_lock);
    _mm_free(all_taskq_locks);
    _mm_free(all_taskqs);

    /* shut down the tasking library */
    concurrent_fin();

    /* destroy the task queues */
    multiq_destroy();

    /* destroy the sync trees */
    synctreepool_destroy();

    /* destroy the task pools and free all tasks */
    taskpools_destroy();
}


/*  partr_coro() -- coroutine entry point
 */
static void partr_coro(struct concurrent_ctx *ctx)
{
    ptask_t *task = ctx_get_user_ptr(ctx);
    task->result = task->f(task->arg, task->start, task->end);

    /* grain tasks must synchronize */
    if (task->grain_num >= 0) {
        int was_last = 0;

        /* reduce... */
        if (task->red) {
            task->result = reduce(task->arr, task->red, task->rf,
                                  task->result, task->grain_num);
            /*  if this task is last, set the result in the parent task */
            if (task->result) {
                task->parent->red_result = task->result;
                was_last = 1;
            }
        }
        /* ... or just sync */
        else {
            if (last_arriver(task->arr, task->grain_num))
                was_last = 1;
        }

        /* the last task to finish needs to finish up the loop */
        if (was_last) {
            /* a non-parent task must wake up the parent */
            if (task->grain_num > 0) {
                multiq_insert(task->parent, 0);
            }
            /* the parent task was last; it can just end */
        }
        else {
            /* the parent task needs to wait */
            if (task->grain_num == 0) {
                yield_value(task->ctx, (void *)yield_from_sync);
            }
        }
    }
}


/*  setup_task() -- allocate and initialize a task
 */
static ptask_t *setup_task(void *(*f)(void *, int64_t, int64_t), void *arg,
        int64_t start, int64_t end)
{
    ptask_t *task = task_alloc();
    if (task == NULL)
        return NULL;

    ctx_construct(task->ctx, task->stack, TASK_STACK_SIZE, partr_coro, task);
    task->f = f;
    task->arg = arg;
    task->start = start;
    task->end = end;
    task->sticky_tid = -1;
    task->grain_num = -1;

    return task;
}


/*  release_task() -- destroy the coroutine context and free the task
 */
static void *release_task(ptask_t *task)
{
    void *result = task->result;
    ctx_destruct(task->ctx);
    if (task->grain_num == 0  &&  task->red)
        reducer_free(task->red);
    if (task->grain_num == 0  &&  task->arr)
        arriver_free(task->arr);
    task->f = NULL;
    task->arg = task->result = task->red_result = NULL;
    task->start = task->end = 0;
    task->rf = NULL;
    task->parent = task->cq = NULL;
    task->arr = NULL;
    task->red = NULL;
    task_free(task);
    return result;
}


/*  add_to_taskq() -- add the specified task to the sticky task queue
 */
static void add_to_taskq(ptask_t *task)
{
    assert(task->sticky_tid != -1);

    ptask_t **q = all_taskqs[task->sticky_tid];
    int8_t *lock = all_taskq_locks[task->sticky_tid];

    while (__atomic_test_and_set(lock, __ATOMIC_ACQUIRE))
        cpu_pause();

    if (*q == NULL)
        *q = task;
    else {
        ptask_t *pt = *q;
        while (pt->next)
            pt = pt->next;
        pt->next = task;
    }

    __atomic_clear(lock, __ATOMIC_RELEASE);
}


/*  get_from_taskq() -- pop the first task off the sticky task queue
 */
static ptask_t *get_from_taskq()
{
    /* racy check for quick path */
    if (*taskq == NULL)
        return NULL;

    while (__atomic_test_and_set(taskq_lock, __ATOMIC_ACQUIRE))
        cpu_pause();

    ptask_t *task = *taskq;
    if (task) {
        *taskq = task->next;
        task->next = NULL;
    }

    __atomic_clear(taskq_lock, __ATOMIC_RELEASE);

    return task;
}


/*  run_next() -- get the next available task and run it
 */
static int run_next()
{
    ptask_t *task;
    
    /* first check for sticky tasks */
    task = get_from_taskq();

    /* no sticky tasks, go to the multiq */
    if (task == NULL) {
        task = multiq_deletemin();
        if (task == NULL)
            return 0;

        /* terminate tasks tell the thread to die */
        if (task->settings & TASK_TERMINATE) {
            release_task(task);
            return 1;
        }

        /* a sticky task will only come out of the multiq if it has not been run */
        if (task->settings & TASK_IS_STICKY) {
            assert(task->sticky_tid == -1);
            task->sticky_tid = tid;
        }
    }

    /* run/resume the task */
    curr_task = task;
    int64_t y = (int64_t)resume(task->ctx);
    curr_task = NULL;

    /* if the task isn't done, it is either in a CQ, or must be re-queued */
    if (!ctx_is_done(task->ctx)) {
        /* the yield value tells us if the task is in a CQ */
        if (y != yield_from_sync) {
            /* sticky tasks go to the thread's sticky queue */
            if (task->settings & TASK_IS_STICKY)
                add_to_taskq(task);
            /* all others go back into the multiq */
            else
                multiq_insert(task, task->prio);
        }
        return 0;
    }

    /* The task completed. As detached tasks cannot be synced, clean
       those up here.
     */
    if (task->settings & TASK_IS_DETACHED) {
        release_task(task);
        return 0;
    }

    /* add back all the tasks in this one's completion queue */
    while (__atomic_test_and_set(&task->cq_lock, __ATOMIC_ACQUIRE))
        cpu_pause();
    ptask_t *cqtask, *cqnext;
    cqtask = task->cq;
    task->cq = NULL;
    while (cqtask) {
        cqnext = cqtask->next;
        cqtask->next = NULL;
        if (cqtask->settings & TASK_IS_STICKY)
            add_to_taskq(cqtask);
        else
            multiq_insert(cqtask, cqtask->prio);
        cqtask = cqnext;
    }
    __atomic_clear(&task->cq_lock, __ATOMIC_RELEASE);

    return 0;
}


/*  partr_start() -- the runtime entry point

    To be called from thread 0, before creating any tasks. Wraps into
    a task and invokes `f(arg)`; tasks should only be spawned/synced
    from within tasks.
 */
int partr_start(void **ret, void *(*f)(void *, int64_t, int64_t),
        void *arg, int64_t start, int64_t end)
{
    assert(tid == 0);

    start_task = setup_task(f, arg, start, end);
    if (start_task == NULL)
        return -1;
    start_task->settings |= TASK_IS_STICKY;
    start_task->sticky_tid = tid;

    curr_task = start_task;
    int64_t y = (int64_t)resume(start_task->ctx);
    curr_task = NULL;

    if (!ctx_is_done(start_task->ctx)) {
        if (y != yield_from_sync) {
            add_to_taskq(start_task);
        }
        while (run_next() == 0)
            if (ctx_is_done(start_task->ctx))
                break;
    }

    void *r = release_task(start_task);
    if (ret)
        *ret = r;

    return 0;
}


/*  partr_thread() -- the thread function

    Loops, getting tasks from the multiqueue and executing them.
 */
static void *partr_thread(void *arg_)
{
    BARRIER_THREAD_DECL;
    lthread_arg_t *arg = (lthread_arg_t *)arg_;

    tid = arg->tid;
    seed_cong(&rngseed);

    /* set affinity if requested */
    if (arg->exclusive) {
        hwloc_set_cpubind(arg->topology, arg->cpuset, HWLOC_CPUBIND_THREAD);
        hwloc_bitmap_free(arg->cpuset);
    }
    show_affinity();

    /* allocate this thread's sticky task queue pointer and initialize the lock */
    taskq_lock = (int8_t *)_mm_malloc(sizeof(int8_t) + sizeof(ptask_t *), 64);
    taskq = (ptask_t **)(taskq_lock + sizeof(int8_t));
    __atomic_clear(taskq_lock, __ATOMIC_RELAXED);
    *taskq = NULL;
    all_taskqs[tid] = taskq;
    all_taskq_locks[tid] = taskq_lock;

    BARRIER();

    /* free the thread function argument */
    free(arg);

    /* get the highest priority task and run it */
    while (run_next() == 0)
        ;

    /* free the sticky task queue pointer (and its lock) */
    _mm_free(taskq_lock);

    return NULL;
}


/*  partr_spawn() -- create a task for `f(arg)` and enqueue it for execution

    Implicitly asserts that `f(arg)` can run concurrently with everything
    else that's currently running. If `detach` is set, the spawned task
    will not be returned (and cannot be synced). Yields.
 */
int partr_spawn(partr_t *t, void *(*f)(void *, int64_t, int64_t),
        void *arg, int64_t start, int64_t end, int8_t sticky, int8_t detach)
{
    ptask_t *task = setup_task(f, arg, start, end);
    if (task == NULL)
        return -1;
    if (sticky)
        task->settings |= TASK_IS_STICKY;
    if (detach)
        task->settings |= TASK_IS_DETACHED;

    if (multiq_insert(task, tid) != 0) {
        release_task(task);
        return -2;
    }

    *t = detach ? NULL : (partr_t)task;

    /* only yield if we're running a non-sticky task */
    if (!(curr_task->settings & TASK_IS_STICKY))
        yield(curr_task->ctx);

    return 0;
}


/*  partr_sync() -- get the return value of task `t`

    Returns only when task `t` has completed.
 */
int partr_sync(void **r, partr_t t, int done_with_task)
{
    ptask_t *task = (ptask_t *)t;

    /* if the target task has not finished, add the current task to its
       completion queue; the thread that runs the target task will add
       this task back to the ready queue
     */
    if (!ctx_is_done(task->ctx)) {
        curr_task->next = NULL;
        while (__atomic_test_and_set(&task->cq_lock, __ATOMIC_ACQUIRE))
            cpu_pause();

        /* ensure the task didn't finish before we got the lock */
        if (!ctx_is_done(task->ctx)) {
            /* add the current task to the CQ */
            if (task->cq == NULL)
                task->cq = curr_task;
            else {
                ptask_t *pt = task->cq;
                while (pt->next)
                    pt = pt->next;
                pt->next = curr_task;
            }

            /* unlock the CQ and yield the current task */
            __atomic_clear(&task->cq_lock, __ATOMIC_RELEASE);
            yield_value(curr_task->ctx, (void *)yield_from_sync);
        }

        /* the task finished before we could add to its CQ */
        else {
            __atomic_clear(&task->cq_lock, __ATOMIC_RELEASE);
        }
    }

    if (r)
        *r = task->grain_num >= 0 && task->red ?
                task->red_result : task->result;

    if (done_with_task)
        release_task(task);

    return 0;
}


/*  partr_parfor() -- spawn multiple tasks for a parallel loop

    Spawn tasks that invoke `f(arg, start, end)` such that the sum of `end-start`
    for all tasks is `count`. Uses `rf()`, if provided, to reduce the return
    values from the tasks, and returns the result. Yields.
 */
int partr_parfor(partr_t *t, void *(*f)(void *, int64_t, int64_t),
        void *arg, int64_t count, void *(*rf)(void *, void *))
{
    int64_t n = GRAIN_K * nthreads;
    lldiv_t each = lldiv(count, n);

    /* allocate synchronization tree(s) */
    arriver_t *arr = arriver_alloc();
    if (arr == NULL)
        return -1;
    reducer_t *red = NULL;
    if (rf != NULL) {
        red = reducer_alloc();
        if (red == NULL) {
            arriver_free(arr);
            return -2;
        }
    }

    /* allocate and enqueue (GRAIN_K * nthreads) tasks */
    *t = NULL;
    int64_t start = 0, end;
    for (int64_t i = 0;  i < n;  ++i) {
        end = start + each.quot + (i < each.rem ? 1 : 0);
        ptask_t *task = setup_task(f, arg, start, end);
        if (task == NULL)
            return -1;

        /* The first task is the parent (root) task of the parfor, thus only
           this can be synced. So, we create the remaining tasks detached.
         */
        if (*t == NULL) *t = task;
        else task->settings = TASK_IS_DETACHED;

        task->parent = *t;
        task->grain_num = i;
        task->rf = rf;
        task->arr = arr;
        task->red = red;

        if (multiq_insert(task, tid) != 0) {
            release_task(task);
            return -3;
        }

        start = end;
    }

    /* only yield if we're running a non-sticky task */
    if (!(curr_task->settings & TASK_IS_STICKY))
        yield(curr_task->ctx);

    return 0;
}

