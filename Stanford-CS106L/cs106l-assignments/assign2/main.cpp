/*
 * CS106L Assignment 2: Marriage Pact
 * Created by Haven Whitney with modifications by Fabio Ibanez & Jacob
 * Roberts-Baca.
 *
 * Welcome to Assignment 2 of CS106L! Please complete each STUDENT TODO
 * in this file. You do not need to modify any other files.
 *
 */

#include <fstream>
#include <iostream>
#include <queue>
#include <set>
#include <string>
// #include <unordered_set>

// std::string kYourName = "Gentaro Watanabe";  // Don't forget to change this!
std::string kYourName = "Mosen Pu";

/**
 * Takes in a file name and returns a set containing all of the applicant
 * names as a set.
 *
 * @param filename  The name of the file to read.
 *                  Each line of the file will be a single applicant's name.
 * @returns         A set of all applicant names read from the file.
 *
 * @remark Feel free to change the return type of this function (and the
 * function below it) to use a `std::unordered_set` instead. If you do so,
 * make sure to also change the corresponding functions in `utils.h`.
 */
std::set<std::string> get_applicants(std::string filename) {
    // STUDENT TODO: Implement this function.
    std::set<std::string> applicants;
    std::ifstream ifs(filename);
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            applicants.insert(line);
            std::cout << line << " ";
        }
        ifs.close();
    }
    return applicants;
}

/**
 * Takes in a set of student names by reference and returns a queue of names
 * that match the given student name.
 *
 * @param name      The returned queue of names should have the same initials as
 * this name.
 * @param students  The set of student names.
 * @return          A queue containing pointers to each matching name.
 */

bool is_initials_match(std::string name, std::string candidate) {
    auto iter_name_2 = name.begin();
    auto iter_cand_2 = candidate.begin();
    while (*iter_name_2 != ' ' || *iter_cand_2 != ' ') {
        if (*iter_name_2 != ' ') iter_name_2++;
        if (*iter_cand_2 != ' ') iter_cand_2++;
    }
    auto iter_name = name.begin();
    auto iter_cand = candidate.begin();
    return (*(++iter_name_2) == *(++iter_cand_2) && *iter_name == *iter_cand);
}

std::queue<const std::string *> find_matches(std::string name,
                                             std::set<std::string> &students) {
    // STUDENT TODO: Implement this function.
    std::queue<const std::string *> matches;
    for (auto iter = students.begin(); iter != students.end(); ++iter) {
        if (is_initials_match(name, *iter)) {
            matches.push(&(*iter));
        }
    }
    return matches;
}

/**
 * Takes in a queue of pointers to possible matches and determines the one
 * true match!
 *
 * You can implement this function however you'd like, but try to do
 * something a bit more complicated than a simple `pop()`.
 *
 * @param matches The queue of possible matches.
 * @return        Your magical one true love.
 *                Will return "NO MATCHES FOUND." if `matches` is empty.
 */
std::string get_match(std::queue<const std::string *> &matches) {
    // STUDENT TODO: Implement this function.
    if (matches.empty()) {
        return "NO MATCHES FOUND.";
    } else {
        return *matches.front();
    }
}

/* #### Please don't remove this line! #### */
#include "autograder/utils.hpp"
