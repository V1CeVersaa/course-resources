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
#include <sstream>
#include <string>
#include <unordered_set>

#include "utils.h"

std::string kYourName = "Nagasaki Soyo"; // Don't forget to change this!

/**
 * Takes in a file name and returns a set containing all of the applicant names
 * as a set.
 *
 * @param filename  The name of the file to read.
 *                  Each line of the file will be a single applicant's name.
 * @returns         A set of all applicant names read from the file.
 *
 * @remark Feel free to change the return type of this function (and the
 * function below it) to use a `std::unordered_set` instead. If you do so, make
 * sure to also change the corresponding functions in `utils.h`.
 */
std::set<std::string> get_applicants(std::string filename) {
  std::set<std::string> applicants;
  std::ifstream ifs(filename);
  if (ifs.is_open()) {
    std::string line;
    while (std::getline(ifs, line)) {
      applicants.insert(line);
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

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> return_vec;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, delim)) {
    return_vec.push_back(token);
  }
  return return_vec;
}

std::queue<const std::string *> find_matches(std::string name,
                                             std::set<std::string> &students) {
  std::queue<const std::string *> matches;
  auto name_vec = split(name, ' ');
  for (const std::string &student_name : students) {
    auto student_name_vec = split(student_name, ' ');
    if (name_vec[0][0] == student_name_vec[0][0] &&
        name_vec[1][0] == student_name_vec[1][0]) {
      matches.push(&student_name);
    }
  }
  return matches;
}

/**
 * Takes in a queue of pointers to possible matches and determines the one true
 * match!
 *
 * You can implement this function however you'd like, but try to do something a
 * bit more complicated than a simple `pop()`.
 *
 * @param matches The queue of possible matches.
 * @return        Your magical one true love.
 *                Will return "NO MATCHES FOUND." if `matches` is empty.
 */
std::string get_match(std::queue<const std::string *> &matches) {
  if (matches.empty()) {
    return "NO MATCHES FOUND.";
  }

  std::string match_name = *matches.front();
  matches.pop();

  for (int i = 0; i < matches.size() - 1; i++) {
    std::string name = *matches.front();
    matches.pop();
    if (name.size() < match_name.size()) {
      match_name = name;
    }
  }
  return match_name;
}

/* #### Please don't modify this call to the autograder! #### */
int main() { return run_autograder(); }
