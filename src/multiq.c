// This file is a part of Julia. License is MIT: https://julialang.org/license

/*  partr -- parallel tasks runtime

    MultiQueues (http://arxiv.org/abs/1411.1209)
 */


#include "julia.h"
#include "julia_internal.h"
#include "partr.h"

#include <stdlib.h>


typedef struct taskheap_tag {
    jl_mutex_t lock;
    ptask_t **tasks;
    int16_t ntasks, prio;
} taskheap_t;

static const int16_t heap_d = 8;

static taskheap_t *heaps;
static int16_t heap_p;

/* unbias state for the RNG */
static uint64_t cong_unbias;


/*  multiq_init()
 */
void multiq_init()
{
    heap_p = MULTIQ_HEAP_C * jl_n_threads;
    heaps = (taskheap_t *)calloc(heap_p, sizeof(taskheap_t));
    for (int16_t i = 0;  i < heap_p;  ++i) {
        jl_mutex_init(&heaps[i].lock);
        heaps[i].tasks = (ptask_t **)
                calloc(MULTIQ_TASKS_PER_HEAP, sizeof(ptask_t *));
        heaps[i].ntasks = 0;
        heaps[i].prio = INT16_MAX;
    }
    unbias_cong(heap_p, &cong_unbias);
}


/*  sift_up()
 */
static void sift_up(taskheap_t *heap, int16_t idx)
{
    if (idx > 0) {
        int16_t parent = (idx-1)/heap_d;
        if (heap->tasks[idx]->prio <= heap->tasks[parent]->prio) {
            ptask_t *t = heap->tasks[parent];
            heap->tasks[parent] = heap->tasks[idx];
            heap->tasks[idx] = t;
            sift_up(heap, parent);
        }
    }
}


/*  sift_down()
 */
void sift_down(taskheap_t *heap, int16_t idx)
{
    if (idx < heap->ntasks) {
        for (int16_t child = heap_d*idx + 1;
                child < MULTIQ_TASKS_PER_HEAP && child <= heap_d*idx + heap_d;
                ++child) {
            if (heap->tasks[child]
                    &&  heap->tasks[child]->prio <= heap->tasks[idx]->prio) {
                ptask_t *t = heap->tasks[idx];
                heap->tasks[idx] = heap->tasks[child];
                heap->tasks[child] = t;
                sift_down(heap, child);
            }
        }
    }
}


/*  multiq_insert()
 */
int multiq_insert(ptask_t *task, int16_t priority)
{
    uint64_t rn;
    
    task->prio = priority;
    do {
        rn = cong(heap_p, cong_unbias, &rngseed);
    } while (!jl_mutex_trylock_nogc(&heaps[rn].lock));

    if (heaps[rn].ntasks >= MULTIQ_TASKS_PER_HEAP) {
        jl_mutex_unlock_nogc(&heaps[rn].lock);
        return -1;
    }

    heaps[rn].tasks[heaps[rn].ntasks++] = task;
    sift_up(&heaps[rn], heaps[rn].ntasks-1);
    jl_mutex_unlock_nogc(&heaps[rn].lock);
    int16_t prio = jl_atomic_load(&heaps[rn].prio);
    if (task->prio < prio)
        jl_atomic_compare_exchange(&heaps[rn].prio, prio, task->prio);

    return 0;
}


/*  multiq_deletemin()
 */
ptask_t *multiq_deletemin()
{
    uint64_t rn1, rn2;
    int16_t i, prio1, prio2;
    ptask_t *task;

    for (i = 0;  i < jl_n_threads;  ++i) {
        rn1 = cong(heap_p, cong_unbias, &rngseed);
        rn2 = cong(heap_p, cong_unbias, &rngseed);
        prio1 = jl_atomic_load(&heaps[rn1].prio);
        prio2 = jl_atomic_load(&heaps[rn2].prio);
        if (prio1 > prio2) {
            prio1 = prio2;
            rn1 = rn2;
        }
        else if (prio1 == prio2 && prio1 == INT16_MAX)
            continue;
        if (jl_mutex_trylock_nogc(&heaps[rn1].lock)) {
            if (prio1 == heaps[rn1].prio)
                break;
            jl_mutex_unlock_nogc(&heaps[rn1].lock);
        }
    }
    if (i == jl_n_threads)
        return NULL;

    task = heaps[rn1].tasks[0];
    heaps[rn1].tasks[0] = heaps[rn1].tasks[--heaps[rn1].ntasks];
    heaps[rn1].tasks[heaps[rn1].ntasks] = NULL;
    prio1 = INT16_MAX;
    if (heaps[rn1].ntasks > 0) {
        sift_down(&heaps[rn1], 0);
        prio1 = heaps[rn1].tasks[0]->prio;
    }
    jl_atomic_store(&heaps[rn1].prio, prio1);
    jl_mutex_unlock_nogc(&heaps[rn1].lock);

    return task;
}


/*  multiq_minprio()
 */
int16_t multiq_minprio()
{
    uint64_t rn1, rn2;
    int16_t prio1, prio2;

    rn1 = cong(heap_p, cong_unbias, &rngseed);
    rn2 = cong(heap_p, cong_unbias, &rngseed);
    prio1 = jl_atomic_load(&heaps[rn1].prio);
    prio2 = jl_atomic_load(&heaps[rn2].prio);
    if (prio2 < prio1)
        return prio2;
    return prio1;
}

