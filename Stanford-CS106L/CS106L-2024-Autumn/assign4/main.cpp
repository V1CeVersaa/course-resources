/*
 * CS106L Assignment 4: Weather Forecast
 * Created by Haven Whitney with modifications by Fabio Ibanez & Jacob Roberts-Baca.
 */

#include <algorithm>
#include <random>
#include <vector>
#include <iostream>


/* #### Please feel free to use these values, but don't change them! #### */
double kMaxTempRequirement = 5;
double uAvgTempRequirement = 60;

struct Forecast {
  double min_temp;
  double max_temp;
  double avg_temp;
};

Forecast compute_forecast(const std::vector<double>& dailyWeather) {
    double min_temp { *std::min_element(dailyWeather.begin(), dailyWeather.end()) };
    double max_temp { *std::max_element(dailyWeather.begin(), dailyWeather.end()) };
    double avg_temp { std::accumulate(dailyWeather.begin(), dailyWeather.end(), static_cast<double>(0.0)) / static_cast<double>(dailyWeather.size()) };

    return {min_temp, max_temp, avg_temp};
}

std::vector<Forecast> get_forecasts(const std::vector<std::vector<double>>& weatherData) {
    std::vector<Forecast> forecasts;
    
    std::transform(weatherData.begin(), weatherData.end(), std::back_inserter(forecasts), compute_forecast);
    
    return forecasts;
}

std::vector<Forecast> get_filtered_data(const std::vector<Forecast>& forecastData) {
    std::vector<Forecast> filteredData {forecastData};

    auto filter {
        [] (const auto& forecast) {
            return (forecast.max_temp - forecast.min_temp <= kMaxTempRequirement) || (forecast.avg_temp < uAvgTempRequirement);
        }
    };

    filteredData.erase(std::remove_if(filteredData.begin(), filteredData.end(), filter), filteredData.end());

    return filteredData;
}


std::vector<Forecast> get_shuffled_data(const std::vector<Forecast>& forecastData) {
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<Forecast> shuffledData {forecastData};

    std::shuffle(shuffledData.begin(), shuffledData.end(), g);

    return shuffledData;
}

std::vector<Forecast> run_weather_pipeline(const std::vector<std::vector<double>>& weatherData) {
    std::vector<Forecast> forecasts {get_forecasts(weatherData)};
    std::vector<Forecast> filteredData {get_filtered_data(forecasts)};
    std::vector<Forecast> shuffledData {get_shuffled_data(filteredData)};
    return shuffledData;
}

/* #### Please don't change this line! #### */
#include "utils.cpp"