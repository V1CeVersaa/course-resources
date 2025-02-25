/*
 * CS106L Assignment 3: Make a Class
 * Created by Fabio Ibanez with modifications by Jacob Roberts-Baca.
 */

#include <iostream>
#include <string>
#include "class.h"

/* #### Please don't change this line! #### */
int run_autograder();



int main() {
    Student xes("xestray", 20, "ZJU");
        
    std::cout << "Name: " << xes.get_name() << "\nAge: " << xes.get_age() << "\nSchool: " << xes.get_school() << "\n";

    xes.haveBirthday();

    /* #### Please don't change this line! #### */
    return run_autograder();
}

/* #### Please don't change this line! #### */
#include "utils.hpp"