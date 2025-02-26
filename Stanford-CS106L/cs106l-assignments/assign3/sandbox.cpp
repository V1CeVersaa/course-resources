/*
 * CS106L Assignment 3: Make a Class
 * Created by Fabio Ibanez with modifications by Jacob Roberts-Baca.
 */

#include <iostream>
#include "class.h"

void sandbox() {
    // STUDENT TODO: Construct an instance of your class!
    Student Mosen("Mosen", 20, "ZJU");
    std::cout << "Name: " << Mosen.getName() << "\nAge: " << Mosen.getAge()
              << "\nSchool: " << Mosen.get_school() << "\n";

    Mosen.haveBirthday();
}