#include "user.h"

/**
 * Creates a new User with the given name and no friends.
 */
User::User(const std::string &name) : _name(name), _friends(nullptr), _size(0), _capacity(0) {}

/**
 * Adds a friend to this User's list of friends.
 * @param name The name of the friend to add.
 */
void User::add_friend(const std::string &name) {
    if (_size == _capacity) {
        _capacity = 2 * _capacity + 1;
        std::string *newFriends = new std::string[_capacity];
        for (size_t i = 0; i < _size; ++i) {
            newFriends[i] = _friends[i];
        }
        delete[] _friends;
        _friends = newFriends;
    }

    _friends[_size++] = name;
}

/**
 * Returns the name of this User.
 */
std::string User::get_name() const { return _name; }

/**
 * Returns the number of friends this User has.
 */
size_t User::size() const { return _size; }

/**
 * Sets the friend at the given index to the given name.
 * @param index The index of the friend to set.
 * @param name The name to set the friend to.
 */
void User::set_friend(size_t index, const std::string &name) { _friends[index] = name; }

/**
 * STUDENT TODO:
 * The definitions for your custom operators and special member functions will go here!
 */

/**
 * Outputs a string representation of a User to the given output stream.
 * Format: User(name=<name>, friends=[<friend1>, <friend2>, ...])
 * @param os The output stream to write to.
 * @param user The User to output.
 * @return The output stream.
 */
std::ostream &operator<<(std::ostream &os, const User &user) {
    os << "User(name=" << user._name << ", friends=[";

    for (size_t i = 0; i < user._size; ++i) {
        if (i > 0) os << ", ";
        os << user._friends[i];
    }
    os << "])";
    return os;
}

/**
 * Destructor for User class.
 * Frees the memory allocated for the _friends array.
 */
User::~User() { delete[] _friends; }

/**
 * Copy constructor for User class.
 * Creates a deep copy of the given User.
 * @param user The User to copy.
 */
User::User(const User &user) {
    _name = user._name;
    _size = user._size;
    _capacity = user._capacity;

    if (user._capacity > 0) {
        _friends = new std::string[user._capacity];
        for (size_t i = 0; i < user._size; ++i) {
            _friends[i] = user._friends[i];
        }
    } else
        _friends = nullptr;
}

/**
 * Copy assignment operator for User class.
 * Creates a deep copy of the given User.
 * @param user The User to copy.
 * @return A reference to this User.
 */
User &User::operator=(const User &user) {
    if (this == &user) return *this;

    delete[] _friends;

    _name = user._name;
    _size = user._size;
    _capacity = user._capacity;

    if (user._capacity > 0) {
        _friends = new std::string[user._capacity];
        for (size_t i = 0; i < user._size; ++i) {
            _friends[i] = user._friends[i];
        }
    } else
        _friends = nullptr;

    return *this;
}

User &User::operator+=(User &other) {
    this->add_friend(other._name);
    other.add_friend(this->_name);

    return *this;
}

// Use the built-in operator< for std::string to compare the names
bool User::operator<(const User &other) const { return this->_name < other._name; }
