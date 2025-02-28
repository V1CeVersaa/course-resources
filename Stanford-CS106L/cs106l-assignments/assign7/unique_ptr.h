#pragma once

#include <cstddef>
#include <utility>

/**
 * @brief A smart pointer that owns an object and deletes it when it goes out of scope.
 * @tparam T The type of the object to manage.
 * @note This class is a simpler version of `std::unique_ptr`.
 */
template <typename T>
class unique_ptr {
   private:
    /* STUDENT TODO: What data must a unique_ptr keep track of? */
    T *ptr{nullptr};

   public:
    /**
     * @brief Constructs a new `unique_ptr` from the given pointer.
     * @param ptr The pointer to manage.
     * @note You should avoid using this constructor directly and instead use `make_unique()`.
     */
    unique_ptr(T *ptr) : ptr(ptr) {}

    /**
     * @brief Constructs a new `unique_ptr` from `nullptr`.
     */
    unique_ptr(std::nullptr_t) : ptr(nullptr) {}

    /**
     * @brief Constructs an empty `unique_ptr`.
     * @note By default, a `unique_ptr` points to `nullptr`.
     */
    unique_ptr() : unique_ptr(nullptr) {}

    /**
     * @brief Dereferences a `unique_ptr` and returns a reference to the object.
     * @return A reference to the object.
     */
    T &operator*() { return *ptr; }
    /* STUDENT TODO: Implement the dereference operator */

    /**
     * @brief Dereferences a `unique_ptr` and returns a const reference to the object.
     * @return A const reference to the object.
     */
    const T &operator*() const { return *ptr; }
    /* STUDENT TODO: Implement the dereference operator (const) */

    /**
     * @brief Returns a pointer to the object managed by the `unique_ptr`.
     * @note This allows for accessing the members of the managed object through the `->` operator.
     * @return A pointer to the object.
     */
    T *operator->() { return ptr; }
    /* STUDENT TODO: Implement the arrow operator */

    /**
     * @brief Returns a const pointer to the object managed by the `unique_ptr`.
     * @note This allows for accessing the members of the managed object through the `->` operator.
     * @return A const pointer to the object.
     */
    const T *operator->() const { return ptr; }
    /* STUDENT TODO: Implement the arrow operator */

    /**
     * @brief Returns whether or not the `unique_ptr` is non-null.
     * @note This allows us to use a `unique_ptr` inside an if-statement.
     * @return `true` if the `unique_ptr` is non-null, `false` otherwise.
     */
    operator bool() const { return ptr != nullptr; }
    /* STUDENT TODO: Implement the boolean conversion operator */

    /** STUDENT TODO: In the space below, do the following:
     * - Implement a destructor
     * - Delete the copy constructor
     * - Delete the copy assignment operator
     * - Implement the move constructor
     * - Implement the move assignment operator
     */

    /**
     * @brief Destructor that deletes the managed object.
     */
    ~unique_ptr() { delete ptr; }

    /**
     * @brief Copy constructor (deleted).
     */
    unique_ptr(const unique_ptr &other) = delete;

    /**
     * @brief Copy assignment operator (deleted).
     */
    unique_ptr &operator=(const unique_ptr &other) = delete;

    /**
     * @brief Move constructor.
     */
    unique_ptr(unique_ptr &&other) : ptr(other.ptr) { other.ptr = nullptr; }

    /**
     * @brief Move assignment operator.
     */
    unique_ptr &operator=(unique_ptr &&other) {
        if (this != &other) {
            delete ptr;
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
};

/**
 * @brief Creates a new unique_ptr for a type with the given arguments.
 * @example auto ptr = make_unique<int>(5);
 * @tparam T The type to create a unique_ptr for.
 * @tparam Args The types of the arguments to pass to the constructor of T.
 * @param args The arguments to pass to the constructor of T.
 */
template <typename T, typename... Args>
unique_ptr<T> make_unique(Args &&...args) {
    return unique_ptr<T>(new T(std::forward<Args>(args)...));
}
