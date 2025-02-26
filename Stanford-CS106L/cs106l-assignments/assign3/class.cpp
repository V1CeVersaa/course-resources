#include "class.h"

#include <iostream>
#include <string>

Person::Person() {}

Person::Person(const std::string &name, int age) : name(name), age(age) {}

void Person::setName(const std::string &name) { this->name = name; }

std::string Person::getName() const { return name; }

void Person::haveBirthday() {
    increaseAge();
    std::cout << getName() << ", happy birthday! Now you are " << getAge()
              << "!\n";
}

int Person::getAge() const { return age; }

void Person::increaseAge() { ++age; }

const std::string Student::get_school() const { return school; }

void Student::set_school(std::string school) { this->school = school; }
