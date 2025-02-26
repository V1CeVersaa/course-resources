#ifndef CLASS_H
#define CLASS_H

#include <string>

class Person {
   public:
    Person();
    Person(const std::string &name, int age);
    ~Person() = default;

    void setName(const std::string &name);
    std::string getName() const;

    void haveBirthday();
    int getAge() const;

   private:
    std::string name = "???";
    int age = -1;

    void increaseAge();
};

class Student : public Person {
   public:
    Student() = default;
    Student(std::string name, int age, std::string school)
        : Person(name, age), school(school) {}
    ~Student() = default;

    const std::string get_school() const;
    void set_school(std::string school);

   private:
    std::string school = "???";
};

#endif
