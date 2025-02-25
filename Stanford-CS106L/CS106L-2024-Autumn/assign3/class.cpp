#include <string>
#include <iostream>
#include "class.h"

// member function of Person
Person::Person() {}

Person::Person(std::string name, int age)
    : name{name}, age{age}
{
}

void Person::set_name(const std::string& name)
{
    this->name = name;
    return;
}

std::string Person::get_name() const
{
    return name;
}

void Person::increaseAge()
{
    ++age;
    return;
}

void Person::haveBirthday()
{
    increaseAge();
    std::cout << get_name() << ", happy birthday! Now you are " << get_age() << "!\n";
    return;
}

int Person::get_age() const
{
    return age;
}

// member function of Student
Student::Student() {}

Student::Student(std::string name, int age, std::string school)
    : Person(name, age), school{school}
{
}


const std::string Student::get_school() const
{
    return school;
}

void Student::set_school(std::string school)
{
    this->school = school;
}
