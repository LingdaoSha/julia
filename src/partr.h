// This file is a part of Julia. License is MIT: https://julialang.org/license

/*  partr -- parallel tasks runtime options
 */

#ifndef PARTR_H
#define PARTR_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef JULIA_ENABLE_PARTR

#include "julia.h"


/* tasks */
#define TASK_STACK_SIZE                 (1024*4)

/* pools */
#define TASKS_PER_POOL                  1024
    /* number allocated = TASKS_PER_POOL * nthreads */

/* multiq */
#define MULTIQ_HEAP_C                   4
    /* number of heaps = MULTIQ_HEAP_C * nthreads */
#define MULTIQ_TASKS_PER_HEAP           129
    /* how many in each heap */

/* parfor */
#define GRAIN_K                         4
    /* tasks = niters / (GRAIN_K * nthreads) */

/* synchronization */
#define ARRIVERS_P                      2
    /* narrivers = ((GRAIN_K * nthreads) ^ ARRIVERS_P) + 1
       limit for number of recursive parfors */
#define REDUCERS_FRAC                   1
    /* nreducers = narrivers * REDUCERS_FRAC */


extern jl_ptls_t *jl_all_tls_states;    /* thread local storage */
extern JL_DLLEXPORT int jl_n_threads;   /* # threads we're actually using */


/* interface */
typedef void *partr_t;

void partr_init();
void partr_shutdown();
int  partr_start(void **ret, void *(*f)(void *, int64_t, int64_t),
        void *arg, int64_t start, int64_t end);
int  partr_spawn(partr_t *t, void *(*f)(void *, int64_t, int64_t),
        void *arg, int64_t start, int64_t end, int8_t sticky,
        int8_t detach);
int  partr_sync(void **r, partr_t t, int done_with_task);
int  partr_parfor(partr_t *t, void *(*f)(void *, int64_t, int64_t),
        void *arg, int64_t count, void *(*rf)(void *, void *));

#endif /* JULIA_ENABLE_PARTR */

#ifdef __cplusplus
}
#endif

#endif /* PARTR_H */

