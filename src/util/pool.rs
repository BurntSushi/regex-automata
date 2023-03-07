// This module provides a relatively simple thread-safe pool of reusable
// objects. For the most part, it's implemented by a stack represented by a
// Mutex<Vec<T>>. It has one small trick: because unlocking a mutex is somewhat
// costly, in the case where a pool is accessed by the first thread that tried
// to get a value, we bypass the mutex. Here are some benchmarks showing the
// difference.
//
// 2022-10-15: These benchmarks are from the old regex crate and they aren't
// easy to reproduce because some rely on older implementations of Pool that
// are no longer around. I've left the results here for posterity, but any
// enterprising individual should feel encouraged to re-litigate the way Pool
// works. I am not at all certain it is the best approach.
//
// 1) misc::anchored_literal_long_non_match    21 (18571 MB/s)
// 2) misc::anchored_literal_long_non_match   107 (3644 MB/s)
// 3) misc::anchored_literal_long_non_match    45 (8666 MB/s)
// 4) misc::anchored_literal_long_non_match    19 (20526 MB/s)
//
// (1) represents our baseline: the master branch at the time of writing when
// using the 'thread_local' crate to implement the pool below.
//
// (2) represents a naive pool implemented completely via Mutex<Vec<T>>. There
// is no special trick for bypassing the mutex.
//
// (3) is the same as (2), except it uses Mutex<Vec<Box<T>>>. It is twice as
// fast because a Box<T> is much smaller than the T we use with a Pool in this
// crate. So pushing and popping a Box<T> from a Vec is quite a bit faster
// than for T.
//
// (4) is the same as (3), but with the trick for bypassing the mutex in the
// case of the first-to-get thread.
//
// Why move off of thread_local? Even though (4) is a hair faster than (1)
// above, this was not the main goal. The main goal was to move off of
// thread_local and find a way to *simply* re-capture some of its speed for
// regex's specific case. So again, why move off of it? The *primary* reason is
// because of memory leaks. See https://github.com/rust-lang/regex/issues/362
// for example. (Why do I want it to be simple? Well, I suppose what I mean is,
// "use as much safe code as possible to minimize risk and be as sure as I can
// be that it is correct.")
//
// My guess is that the thread_local design is probably not appropriate for
// regex since its memory usage scales to the number of active threads that
// have used a regex, where as the pool below scales to the number of threads
// that simultaneously use a regex. While neither case permits contraction,
// since we own the pool data structure below, we can add contraction if a
// clear use case pops up in the wild. More pressingly though, it seems that
// there are at least some use case patterns where one might have many threads
// sitting around that might have used a regex at one point. While thread_local
// does try to reuse space previously used by a thread that has since stopped,
// its maximal memory usage still scales with the total number of active
// threads. In contrast, the pool below scales with the total number of threads
// *simultaneously* using the pool. The hope is that this uses less memory
// overall. And if it doesn't, we can hopefully tune it somehow.
//
// It seems that these sort of conditions happen frequently
// in FFI inside of other more "managed" languages. This was
// mentioned in the issue linked above, and also mentioned here:
// https://github.com/BurntSushi/rure-go/issues/3. And in particular, users
// confirm that disabling the use of thread_local resolves the leak.
//
// There were other weaker reasons for moving off of thread_local as well.
// Namely, at the time, I was looking to reduce dependencies. And for something
// like regex, maintenance can be simpler when we own the full dependency tree.

/*!
A thread safe memory pool.

The principal type in this module is a [`Pool`]. It main use case is for
holding a thread safe collection of mutable scratch spaces (usually called
`Cache` in this crate) that regex engines need to execute a search. This then
permits sharing the same read-only regex object across multiple threads while
having a quick way of reusing scratch space in a thread safe way. This avoids
needing to re-create the scratch space for every search, which could wind up
being quite expensive.
*/

/// A thread safe pool that works in an `alloc`-only context.
///
/// Getting a value out comes with a guard. When that guard is dropped, the
/// value is automatically put back in the pool.
///
/// A `Pool` impls `Sync` when `T` is `Send` (even if `T` is not `Sync`). This
/// means that `T` can use interior mutability. This is possible because a pool
/// is guaranteed to provide a value to exactly one thread at any time.
///
/// Currently, a pool never contracts in size. Its size is proportional to the
/// maximum number of simultaneous uses. This may change in the future.
///
/// A `Pool` is a particularly useful data structure for this crate because
/// many of the regex engines require a mutable "cache" in order to execute
/// a search. Since regexes themselves tend to be global, the problem is then:
/// how do you get a mutable cache to execute a search? You could:
///
/// 1. Use a `thread_local!`, which requires the standard library and requires
/// that the regex pattern be statically known.
/// 2. Use a `Pool`.
/// 3. Make the cache an explicit dependency in your code and pass it around.
/// 4. Put the cache state in a `Mutex`, but this means only one search can
/// execute at a time.
/// 5. Create a new cache for every search.
///
/// A `thread_local!` is perhaps the best choice if it works for your use case.
/// Putting the cache in a mutex or creating a new cache for every search are
/// perhaps the worst choices. Of the remaining two choices, whether you use
/// this `Pool` or thread through a cache explicitly in your code is a matter
/// of taste and depends on your code architecture.
///
/// # Example
///
/// This example shows how to share a single hybrid regex among multiple
/// threads, while also safely getting exclusive access to a hybrid's
/// [`Cache`](crate::hybrid::regex::Cache) without preventing other searches
/// from running while your thread uses the `Cache`.
///
/// ```
/// use regex_automata::{
///     hybrid::regex::{Cache, Regex},
///     util::{lazy::Lazy, pool::Pool},
///     Match,
/// };
///
/// static RE: Lazy<Regex> =
///     Lazy::new(|| Regex::new("foo[0-9]+bar").unwrap());
/// static CACHE: Lazy<Pool<Cache>> =
///     Lazy::new(|| Pool::new(|| RE.create_cache()));
///
/// let expected = Some(Match::must(0, 3..14));
/// assert_eq!(expected, RE.find(&mut CACHE.get(), b"zzzfoo12345barzzz"));
/// ```
#[derive(Debug)]
pub struct Pool<T, F = fn() -> T>(inner::Pool<T, F>);

impl<T, F> Pool<T, F> {
    /// Create a new pool. The given closure is used to create values in
    /// the pool when necessary.
    pub fn new(create: F) -> Pool<T, F> {
        Pool(inner::Pool::new(create))
    }
}

impl<T: Send, F: Fn() -> T> Pool<T, F> {
    /// Get a value from the pool. The caller is guaranteed to have
    /// exclusive access to the given value.
    ///
    /// Note that there is no guarantee provided about which value in the
    /// pool is returned. That is, calling get, dropping the guard (causing
    /// the value to go back into the pool) and then calling get again is
    /// *not* guaranteed to return the same value received in the first `get`
    /// call.
    pub fn get(&self) -> PoolGuard<'_, T, F> {
        PoolGuard(self.0.get())
    }
}

/// A guard that is returned when a caller requests a value from the pool.
///
/// The purpose of the guard is to use RAII to automatically put the value
/// back in the pool once it's dropped.
#[derive(Debug)]
pub struct PoolGuard<'a, T: Send, F: Fn() -> T>(inner::PoolGuard<'a, T, F>);

impl<'a, T: Send, F: Fn() -> T> core::ops::Deref for PoolGuard<'a, T, F> {
    type Target = T;

    fn deref(&self) -> &T {
        self.0.value()
    }
}

impl<'a, T: Send, F: Fn() -> T> core::ops::DerefMut for PoolGuard<'a, T, F> {
    fn deref_mut(&mut self) -> &mut T {
        self.0.value_mut()
    }
}

#[cfg(feature = "std")]
mod inner {
    use core::{
        cell::UnsafeCell,
        panic::{RefUnwindSafe, UnwindSafe},
        sync::atomic::{AtomicUsize, Ordering},
    };

    use alloc::{boxed::Box, vec, vec::Vec};

    use std::{sync::Mutex, thread_local};

    /// An atomic counter used to allocate thread IDs.
    static COUNTER: AtomicUsize = AtomicUsize::new(1);

    thread_local!(
        /// A thread local used to assign an ID to a thread.
        static THREAD_ID: usize = {
            let next = COUNTER.fetch_add(1, Ordering::Relaxed);
            // SAFETY: We cannot permit the reuse of thread IDs since reusing a
            // thread ID might result in more than one thread "owning" a pool,
            // and thus, permit accessing a mutable value from multiple threads
            // simultaneously without synchronization. The intent of this panic
            // is to be a sanity check. It is not expected that the thread ID
            // space will actually be exhausted in practice.
            //
            // This checks that the counter never wraps around, since atomic
            // addition wraps around on overflow.
            if next == 0 {
                panic!("regex: thread ID allocation space exhausted");
            }
            next
        };
    );

    /// A thread safe pool utilizing std-only features.
    ///
    /// The main difference between this and the simplistic alloc-only pool is
    /// the use of std::sync::Mutex and an "owner thread" optimization that
    /// makes accesses by the owner of a pool faster than all other threads.
    /// This makes the common case of running a regex within a single thread
    /// faster by avoiding mutex unlocking.
    pub(super) struct Pool<T, F> {
        /// A stack of T values to hand out. These are used when a Pool is
        /// accessed by a thread that didn't create it.
        stack: Mutex<Vec<Box<T>>>,
        /// A function to create more T values when stack is empty and a caller
        /// has requested a T.
        create: F,
        /// The ID of the thread that owns this pool. The owner is the thread
        /// that makes the first call to 'get'. When the owner calls 'get', it
        /// gets 'owner_val' directly instead of returning a T from 'stack'.
        /// See comments elsewhere for details, but this is intended to be an
        /// optimization for the common case that makes getting a T faster.
        ///
        /// It is initialized to a value of zero (an impossible thread ID) as a
        /// sentinel to indicate that it is unowned.
        owner: AtomicUsize,
        /// A value to return when the caller is in the same thread that
        /// first called `Pool::get`.
        ///
        /// This is set to None when a Pool is first created, and set to Some
        /// once the first thread calls Pool::get.
        owner_val: UnsafeCell<Option<T>>,
    }

    // SAFETY: Since we want to use a Pool from multiple threads simultaneously
    // behind an Arc, we need for it to be Sync. In cases where T is sync,
    // Pool<T> would be Sync. However, since we use a Pool to store mutable
    // scratch space, we wind up using a T that has interior mutability and is
    // thus itself not Sync. So what we *really* want is for our Pool<T> to by
    // Sync even when T is not Sync (but is at least Send).
    //
    // The only non-sync aspect of a Pool is its 'owner_val' field, which is
    // used to implement faster access to a pool value in the common case of
    // a pool being accessed in the same thread in which it was created. The
    // 'stack' field is also shared, but a Mutex<T> where T: Send is already
    // Sync. So we only need to worry about 'owner_val'.
    //
    // The key is to guarantee that 'owner_val' can only ever be accessed from
    // one thread. In our implementation below, we guarantee this by only
    // returning the 'owner_val' when the ID of the current thread matches the
    // ID of the thread that first called 'Pool::get'. Since this can only ever
    // be one thread, it follows that only one thread can access 'owner_val' at
    // any point in time. Thus, it is safe to declare that Pool<T> is Sync when
    // T is Send.
    //
    // If there is a way to achieve our performance goals using safe code, then
    // I would very much welcome a patch. As it stands, the implementation
    // below tries to balance safety with performance. The case where a Regex
    // is used from multiple threads simultaneously will suffer a bit since
    // getting a value out of the pool will require unlocking a mutex.
    unsafe impl<T: Send, F> Sync for Pool<T, F> {}

    // If T is UnwindSafe, then since we provide exclusive access to any
    // particular value in the pool, it should therefore also be considered
    // RefUnwindSafe. Also, since we use std::sync::Mutex, we get poisoning
    // from it if another thread panics while the lock is held.
    impl<T: UnwindSafe, F> RefUnwindSafe for Pool<T, F> {}

    impl<T, F> Pool<T, F> {
        /// Create a new pool. The given closure is used to create values in
        /// the pool when necessary.
        pub(super) fn new(create: F) -> Pool<T, F> {
            // MSRV(1.63): Mark this function as 'const'. I've arranged the
            // code such that it should "just work." Then mark the public
            // 'Pool::new' method as 'const' too. (The alloc-only Pool::new
            // is already 'const', so that should "just work" too.) The only
            // thing we're waiting for is Mutex::new to be const.
            let owner = AtomicUsize::new(0);
            let owner_val = UnsafeCell::new(None); // init'd on first access
            Pool { stack: Mutex::new(vec![]), create, owner, owner_val }
        }
    }

    impl<T: Send, F: Fn() -> T> Pool<T, F> {
        /// Get a value from the pool. This may block if another thread is also
        /// attempting to retrieve a value from the pool.
        pub(super) fn get(&self) -> PoolGuard<'_, T, F> {
            // Our fast path checks if the caller is the thread that "owns"
            // this pool. Or stated differently, whether it is the first thread
            // that tried to extract a value from the pool. If it is, then we
            // can return a T to the caller without going through a mutex.
            //
            // SAFETY: We must guarantee that only one thread gets access
            // to this value. Since a thread is uniquely identified by the
            // THREAD_ID thread local, it follows that if the caller's thread
            // ID is equal to the owner, then only one thread may receive this
            // value.
            let caller = THREAD_ID.with(|id| *id);
            let owner = self.owner.load(Ordering::Acquire);
            if caller == owner {
                return self.guard_owned();
            }
            self.get_slow(caller, owner)
        }

        /// This is the "slow" version that goes through a mutex to pop an
        /// allocated value off a stack to return to the caller. (Or, if the
        /// stack is empty, a new value is created.)
        ///
        /// If the pool has no owner, then this will set the owner.
        #[cold]
        fn get_slow(
            &self,
            caller: usize,
            owner: usize,
        ) -> PoolGuard<'_, T, F> {
            if owner == 0 {
                // The sentinel 0 value means this pool is not yet owned. We
                // try to atomically set the owner. If we do, then this thread
                // becomes the owner and we can return a guard that represents
                // the special T for the owner.
                let res = self.owner.compare_exchange(
                    0,
                    caller,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                if res.is_ok() {
                    // SAFETY: A successful CAS above implies this thread is
                    // the owner and that this is the only such thread that
                    // can reach here. Thus, there is no data race.
                    unsafe {
                        *self.owner_val.get() = Some((self.create)());
                    }
                    return self.guard_owned();
                }
            }
            let mut stack = self.stack.lock().unwrap();
            let value = match stack.pop() {
                None => Box::new((self.create)()),
                Some(value) => value,
            };
            self.guard_stack(value)
        }

        /// Puts a value back into the pool. Callers don't need to call this.
        /// Once the guard that's returned by 'get' is dropped, it is put back
        /// into the pool automatically.
        fn put(&self, value: Box<T>) {
            let mut stack = self.stack.lock().unwrap();
            stack.push(value);
        }

        /// Create a guard that represents the special owned T.
        fn guard_owned(&self) -> PoolGuard<'_, T, F> {
            PoolGuard { pool: self, value: None }
        }

        /// Create a guard that contains a value from the pool's stack.
        fn guard_stack(&self, value: Box<T>) -> PoolGuard<'_, T, F> {
            PoolGuard { pool: self, value: Some(value) }
        }
    }

    impl<T: core::fmt::Debug, F> core::fmt::Debug for Pool<T, F> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("Pool")
                .field("stack", &self.stack)
                .field("owner", &self.owner)
                .field("owner_val", &self.owner_val)
                .finish()
        }
    }

    /// A guard that is returned when a caller requests a value from the pool.
    #[derive(Debug)]
    pub(super) struct PoolGuard<'a, T: Send, F: Fn() -> T> {
        /// The pool that this guard is attached to.
        pool: &'a Pool<T, F>,
        /// This is None when the guard represents the special "owned" value. In
        /// which case, the value is retrieved from 'pool.owner_val'.
        value: Option<Box<T>>,
    }

    impl<'a, T: Send, F: Fn() -> T> PoolGuard<'a, T, F> {
        /// Return the underlying value.
        pub(super) fn value(&self) -> &T {
            match self.value {
                // SAFETY: This is safe because the only way a PoolGuard gets
                // created for self.value=None is when the current thread
                // corresponds to the owning thread, of which there can only
                // be one. Thus, we are guaranteed to be providing exclusive
                // access here which makes this safe.
                //
                // Also, since 'owner_val' is guaranteed to be initialized
                // before an owned PoolGuard is created, the unchecked unwrap
                // is safe.
                None => unsafe {
                    // MSRV(1.58): Use unwrap_unchecked here.
                    (*self.pool.owner_val.get())
                        .as_ref()
                        .unwrap_or_else(|| core::hint::unreachable_unchecked())
                },
                Some(ref v) => &**v,
            }
        }

        /// Return the underlying value as a mutable borrow.
        pub(super) fn value_mut(&mut self) -> &mut T {
            match self.value {
                // SAFETY: This is safe because the only way a PoolGuard gets
                // created for self.value=None is when the current thread
                // corresponds to the owning thread, of which there can only
                // be one. Thus, we are guaranteed to be providing exclusive
                // access here which makes this safe.
                //
                // Also, since 'owner_val' is guaranteed to be initialized
                // before an owned PoolGuard is created, the unwrap_unchecked
                // is safe.
                None => unsafe {
                    // MSRV(1.58): Use unwrap_unchecked here.
                    (*self.pool.owner_val.get())
                        .as_mut()
                        .unwrap_or_else(|| core::hint::unreachable_unchecked())
                },
                Some(ref mut v) => &mut **v,
            }
        }
    }

    impl<'a, T: Send, F: Fn() -> T> Drop for PoolGuard<'a, T, F> {
        fn drop(&mut self) {
            if let Some(value) = self.value.take() {
                self.pool.put(value);
            }
        }
    }
}

#[cfg(not(feature = "std"))]
mod inner {
    use core::{
        cell::UnsafeCell,
        panic::{RefUnwindSafe, UnwindSafe},
        sync::atomic::{AtomicBool, Ordering},
    };

    use alloc::{boxed::Box, vec, vec::Vec};

    /// A thread safe pool utilizing alloc-only features.
    ///
    /// Unlike the std version, it doesn't seem possible(?) to implement the
    /// "thread owner" optimization because alloc-only doesn't have any concept
    /// of threads. So the best we can do is just a normal stack. This will
    /// increase latency in alloc-only environments.
    pub(super) struct Pool<T, F> {
        /// A stack of T values to hand out. These are used when a Pool is
        /// accessed by a thread that didn't create it.
        stack: Mutex<Vec<Box<T>>>,
        /// A function to create more T values when stack is empty and a caller
        /// has requested a T.
        create: F,
    }

    // If T is UnwindSafe, then since we provide exclusive access to any
    // particular value in the pool, it should therefore also be considered
    // RefUnwindSafe.
    impl<T: UnwindSafe, F> RefUnwindSafe for Pool<T, F> {}

    impl<T, F> Pool<T, F> {
        /// Create a new pool. The given closure is used to create values in
        /// the pool when necessary.
        pub(super) const fn new(create: F) -> Pool<T, F> {
            Pool { stack: Mutex::new(vec![]), create }
        }
    }

    impl<T: Send, F: Fn() -> T> Pool<T, F> {
        /// Get a value from the pool. This may block if another thread is also
        /// attempting to retrieve a value from the pool.
        pub(super) fn get(&self) -> PoolGuard<'_, T, F> {
            let mut stack = self.stack.lock();
            let value = match stack.pop() {
                None => Box::new((self.create)()),
                Some(value) => value,
            };
            PoolGuard { pool: self, value: Some(value) }
        }

        /// Puts a value back into the pool. Callers don't need to call this.
        /// Once the guard that's returned by 'get' is dropped, it is put back
        /// into the pool automatically.
        fn put(&self, value: Box<T>) {
            let mut stack = self.stack.lock();
            stack.push(value);
        }
    }

    impl<T: core::fmt::Debug, F> core::fmt::Debug for Pool<T, F> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("Pool").field("stack", &self.stack).finish()
        }
    }

    /// A guard that is returned when a caller requests a value from the pool.
    #[derive(Debug)]
    pub(super) struct PoolGuard<'a, T: Send, F: Fn() -> T> {
        /// The pool that this guard is attached to.
        pool: &'a Pool<T, F>,
        /// This is None when the guard represents the special "owned" value. In
        /// which case, the value is retrieved from 'pool.owner_val'.
        value: Option<Box<T>>,
    }

    impl<'a, T: Send, F: Fn() -> T> PoolGuard<'a, T, F> {
        /// Return the underlying value.
        pub(super) fn value(&self) -> &T {
            self.value.as_deref().unwrap()
        }

        /// Return the underlying value as a mutable borrow.
        pub(super) fn value_mut(&mut self) -> &mut T {
            self.value.as_deref_mut().unwrap()
        }
    }

    impl<'a, T: Send, F: Fn() -> T> Drop for PoolGuard<'a, T, F> {
        fn drop(&mut self) {
            if let Some(value) = self.value.take() {
                self.pool.put(value);
            }
        }
    }

    /// A spin-lock based mutex. Yes, I have read spinlocks cosnidered
    /// harmful[1], and if there's a reasonable alternative choice, I'll
    /// happily take it.
    ///
    /// I suspect the most likely alternative here is a Treiber stack, but
    /// implementing one correctly in a way that avoids the ABA problem looks
    /// subtle enough that I'm not sure I want to attempt that. But otherwise,
    /// we only need a mutex in order to implement our pool, so if there's
    /// something simpler we can use that works for our `Pool` use case, then
    /// that would be great.
    ///
    /// Note that this mutex does not do poisoning.
    ///
    /// [1]: https://matklad.github.io/2020/01/02/spinlocks-considered-harmful.html
    #[derive(Debug)]
    struct Mutex<T> {
        locked: AtomicBool,
        data: UnsafeCell<T>,
    }

    // SAFETY: Since a Mutex guarantees exclusive access, as long as we can
    // send it across threads, it must also be Sync.
    unsafe impl<T: Send> Sync for Mutex<T> {}

    impl<T> Mutex<T> {
        /// Create a new mutex for protecting access to the given value across
        /// multiple threads simultaneously.
        const fn new(value: T) -> Mutex<T> {
            Mutex {
                locked: AtomicBool::new(false),
                data: UnsafeCell::new(value),
            }
        }

        /// Lock this mutex and return a guard providing exclusive access to
        /// `T`. This blocks if some other thread has already locked this
        /// mutex.
        fn lock(&self) -> MutexGuard<'_, T> {
            while self
                .locked
                .compare_exchange(
                    false,
                    true,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                core::hint::spin_loop();
            }
            // SAFETY: The only way we're here is if we successfully set
            // 'locked' to true, which implies we must be the only thread here
            // and thus have exclusive access to 'data'.
            let data = unsafe { &mut *self.data.get() };
            MutexGuard { locked: &self.locked, data }
        }
    }

    /// A guard that derefs to &T and &mut T. When it's dropped, the lock is
    /// released.
    #[derive(Debug)]
    struct MutexGuard<'a, T> {
        locked: &'a AtomicBool,
        data: &'a mut T,
    }

    impl<'a, T> core::ops::Deref for MutexGuard<'a, T> {
        type Target = T;

        fn deref(&self) -> &T {
            self.data
        }
    }

    impl<'a, T> core::ops::DerefMut for MutexGuard<'a, T> {
        fn deref_mut(&mut self) -> &mut T {
            self.data
        }
    }

    impl<'a, T> Drop for MutexGuard<'a, T> {
        fn drop(&mut self) {
            // Drop means 'data' is no longer accessible, so we can unlock
            // the mutex.
            self.locked.store(false, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use core::panic::{RefUnwindSafe, UnwindSafe};

    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn oibits() {
        fn assert_oitbits<T: Send + Sync + UnwindSafe + RefUnwindSafe>() {}
        assert_oitbits::<Pool<Vec<u32>>>();
        assert_oitbits::<Pool<core::cell::RefCell<Vec<u32>>>>();
    }

    // Tests that Pool implements the "single owner" optimization. That is, the
    // thread that first accesses the pool gets its own copy, while all other
    // threads get distinct copies.
    #[cfg(feature = "std")]
    #[test]
    fn thread_owner_optimization() {
        use std::{cell::RefCell, sync::Arc, vec};

        let pool: Arc<Pool<RefCell<Vec<char>>>> =
            Arc::new(Pool::new(|| RefCell::new(vec!['a'])));
        pool.get().borrow_mut().push('x');

        let pool1 = pool.clone();
        let t1 = std::thread::spawn(move || {
            let guard = pool1.get();
            guard.borrow_mut().push('y');
        });

        let pool2 = pool.clone();
        let t2 = std::thread::spawn(move || {
            let guard = pool2.get();
            guard.borrow_mut().push('z');
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // If we didn't implement the single owner optimization, then one of
        // the threads above is likely to have mutated the [a, x] vec that
        // we stuffed in the pool before spawning the threads. But since
        // neither thread was first to access the pool, and because of the
        // optimization, we should be guaranteed that neither thread mutates
        // the special owned pool value.
        //
        // (Technically this is an implementation detail and not a contract of
        // Pool's API.)
        assert_eq!(vec!['a', 'x'], *pool.get().borrow());
    }
}
