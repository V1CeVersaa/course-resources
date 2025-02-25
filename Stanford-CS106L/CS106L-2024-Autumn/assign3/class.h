#ifndef CLASS_H
#define CLASS_H

#include <string>

class Person {
public:
    Person();
    Person(std::string name, int age);
    ~Person() = default;

    void set_name(const std::string& name); 
    std::string get_name() const;
    
    void haveBirthday();
    int get_age() const;

private:
    std::string name {"???"};
    int age {0};

    void increaseAge();
};

class Student: public Person {
public:
    Student();
    Student(std::string name, int age, std::string school);
    const std::string get_school() const;
    void set_school(std::string school);

private:
    std::string school {"???"};
};

#endif