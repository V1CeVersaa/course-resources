#include "spellcheck.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <ranges>
#include <set>
#include <vector>

template <typename Iterator, typename UnaryPred>
std::vector<Iterator> find_all(Iterator begin, Iterator end, UnaryPred pred);

// Corpus for std::set<Token>

Corpus tokenize(const std::string &source) {
    /* TODO: Implement this method */
    auto space_iters = find_all(source.begin(), source.end(), isspace);
    Corpus tokens;
    std::transform(space_iters.begin(), space_iters.end() - 1, space_iters.begin() + 1,
                   std::inserter(tokens, tokens.end()),
                   [&source](auto begin, auto end) { return Token(source, begin, end); });
    // Remove empty tokens
    std::erase_if(tokens, [](const Token &token) { return token.content.empty(); });
    return tokens;
}

std::set<Mispelling> spellcheck(const Corpus &source, const Dictionary &dictionary) {
    /* TODO: Implement this method */
    namespace rv = std::ranges::views;

    // Step 1: Skip words that are already correctly spelled.
    auto misspellings =
        source | rv::filter([&dictionary](const Token &token) { return !dictionary.contains(token.content); });

    // Step 2: Find one-edit-away words in the dictionary using Damerau-Levenshtein
    auto one_edit_away = misspellings | rv::transform([&dictionary](const Token &token) {
                             auto suggestions_view = dictionary | rv::filter([&token](const std::string &word) {
                                                         return levenshtein(token.content, word) == 1;
                                                     });
                             std::set<std::string> suggestions(suggestions_view.begin(), suggestions_view.end());
                             return Mispelling{token, suggestions};
                         });

    // Step 3: Drop misspellings with no suggestions.
    auto valid_misspellings = one_edit_away | rv::filter([](const Mispelling &m) { return !m.suggestions.empty(); });
    return std::set<Mispelling>(valid_misspellings.begin(), valid_misspellings.end());
}

/* Helper methods */

#include "utils.cpp"