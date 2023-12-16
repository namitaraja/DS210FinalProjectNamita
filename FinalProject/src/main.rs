use std::fs::File;
use std::io::{BufRead, BufReader};
use std::collections::HashMap;
use csv::Reader;
use num::average;
use ndarray::{Array2, Axis};

struct Movie {
    name: String,
    year: String,
    user_ratings: f64,
    metascore: f64,
    gross_income: i32,
    votes: i32,
    genres: String,
}

impl Movie {
    fn from(row: &[String]) -> Self {
        Movie {
            name: row[0].to_string(),
            year: row[1].to_string(),
            user_ratings: row[2].parse().unwrap_or(0.0),
            metascore: row[3].parse().unwrap_or(0.0),
            gross_income: row[4].parse().unwrap_or(0),
            votes: row[5].parse().unwrap_or(0),
            genres: row[6].to_string(),
        }
    }
}

fn main() {
    let file = File::open("data.csv").unwrap();
    let reader = BufReader::new(file);
    let mut csv_reader = Reader::new(reader);

    let mut movies = Vec::new();
    csv_reader.read_row().unwrap();

    for row in csv_reader.records() {
        let mut movie = Movie::new();
        let row = row.unwrap();
        movie.name = row[0].to_string();
        movie.year = row[1].to_string();
        movie.user_ratings = row[2].parse().unwrap_or(0.0);
        movie.metascore = row[3].parse().unwrap_or(0.0);
        movie.gross_income = row[4].parse().unwrap_or(0);
        movie.votes = row[5].parse().unwrap_or(0);
        movie.genres = row[6].to_string();
        movie.stars = row[7].to_string();

        movies.push(movie);
    }
    let user_genre = get_user_input("Enter your preferred genre: ");

    println!("Descriptive statistics:");
    println!("Total movies: {}", movies.len());
    println!("Average user rating: {}", average(movies.iter().map(|m| m.user_ratings)).unwrap_or(0.0));
    println!("Average metascore: {}", average(movies.iter().map(|m| m.metascore)).unwrap_or(0.0));
    println!("Average gross income: {}", average(movies.iter().map(|m| m.gross_income as f64)).unwrap_or(0.0));
    println!("Average votes: {}", average(movies.iter().map(|m| m.votes as f64)).unwrap_or(0.0));

    println!("Genre distribution:");
    println!("Movie recommendation system:");
    let mut recommended_movies = Vec::new();
    for movie in &movies {
        let mut similar_movies = Vec::new();
        for (other_movie, &cluster) in movies.iter().zip(labels.iter()) {
            if cluster == labels[movies.iter().position(|m| m.name == movie.name).unwrap()] {
                similar_movies.push(other_movie.clone());
            }
        }
        recommended_movies.push((movie.clone(), similar_movies));
    }

    println!("Evaluating recommendation system...");
    let mut accuracy = 0;
    for (movie, similar_movies) in recommended_movies.iter() {
        let user_rating = movie.user_ratings;
        let similar_ratings = average(similar_movies.iter().map(|m| m.user_ratings)).unwrap_or(0.0);
        if user_rating > similar_ratings {
            accuracy += 1;
        }
    }
    println!("Accuracy: {}%", accuracy as f64 * 100.0 / recommended_movies.len() as f64);
}
    let filtered_movies: Vec<&Movie> = movies
        .iter()
        .filter(|&m| m.genres.contains(&user_genre))
        .collect();

    if !filtered_movies.is_empty() {
        let recommendations = recommend_movies(&filtered_movies, &movies);
        println!("Recommended Movies:");
        for (movie, _) in recommendations {
            println!("{}", movie.name);
        }
    }
    else {
        println!("No movies found with the specified genre.");
    }

fn get_user_input(prompt: &str) -> String {
    println!("{}", prompt);
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).expect("Failed to read input");
    input.trim().to_string()
}

fn recommend_movies(filtered_movies: &[&Movie], all_movies: &[Movie]) -> Vec<(Movie, usize)> {
    const K: usize = 5;
    let clusters = k_means_clustering(all_movies, K);

    let filtered_clusters: Vec<usize> = filtered_movies
        .iter()
        .map(|&m| clusters[m.id])
        .collect();

    let most_common_cluster = mode(&filtered_clusters);

    let recommended_movies: Vec<(Movie, usize)> = all_movies
        .iter()
        .filter(|&m| clusters[m.id] == most_common_cluster)
        .map(|&m| (m.clone(), clusters[m.id]))
        .collect();

    recommended_movies
}

fn k_means_clustering(movies: &Vec<Movie>) -> Vec<usize> {
    let data: Vec<Vec<f64>> = movies
        .iter()
        .map(|m| vec![m.user_ratings, m.metascore, m.gross_income as f64, m.votes as f64])
        .collect();

    let genre_map: HashMap<&str, usize> = movies
        .iter()
        .flat_map(|m| m.genres.split(','))
        .unique()
        .enumerate()
        .map(|(i, genre)| (genre, i))
        .collect();

    let data_array = Array2::from_shape_fn((movies.len(), 5), |(i, j)| {
        if j == 4 {
            *genre_map.get(movies[i].genres.split(',').next().unwrap()).unwrap() as f64
        } else {
            data[i][j]
        }
    });

    let num_clusters = K;
    let tolerance = 1e-5;
    let max_iters = 100;

    let mut centroids = Array2::from_shape_fn((num_clusters, 5), |(_, j)| data_array[[0, j]]);

    let mut labels = vec![0; movies.len()];

    for _ in 0..max_iters {
        let distances = centroids
            .axis_iter(Axis(0))
            .map(|centroid| {
                data_array
                    .view()
                    .map(|x| (x - &centroid).mapv(|v| v * v).sum())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        labels = distances
            .iter()
            .map(|d| d.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0)
            .collect();

        let new_centroids = centroids
            .axis_iter_mut(Axis(0))
            .enumerate()
            .map(|(i, mut centroid)| {
                let members = labels
                    .iter()
                    .enumerate()
                    .filter(|(_, &label)| label == i)
                    .map(|(j, _)| data_array.row(j).to_owned())
                    .collect::<Array2<_>>();

                centroid.assign(&members.mean_axis(Axis(0)).unwrap());

                centroid.to_owned()
            })
            .collect::<Vec<_>>();

        let diff = centroids.mapv_into(|x| 0.0)
            - new_centroids.iter().fold(
                Array2::from_shape_fn((num_clusters, 5), |(_, _)| 0.0),
                |acc, x| acc + x,
            )
            .mapv(|x| x / num_clusters as f64);

        if diff.mapv(|x| x.abs()).sum() < tolerance {
            break;
        }

        centroids = Array2::from_shape_fn((num_clusters, 5), |(i, j)| new_centroids[i][j]);
    }

    labels
}

mod tests {
    use super::*;

    #[test]
    fn test_recommend_movies() {
        let movie1 = Movie::new("Movie1".to_string(), 2022, 8.0, 75.0, 1000000, 500, "Drama".to_string());
        let movie2 = Movie::new("Movie2".to_string(), 2022, 7.5, 80.0, 800000, 400, "Comedy".to_string());
        let movie3 = Movie::new("Movie3".to_string(), 2022, 9.0, 90.0, 1200000, 600, "Action".to_string());

        let all_movies = vec![movie1.clone(), movie2.clone(), movie3.clone()];
        let filtered_movies = vec![&movie1, &movie2];

        let recommendations = recommend_movies(&filtered_movies, &all_movies);

        for (recommended_movie, cluster) in recommendations.iter() {
            assert_eq!(cluster, &recommendations[0].1);
            assert_ne!(recommended_movie, &filtered_movies[0]);
            assert_ne!(recommended_movie, &filtered_movies[1]);
        }
    }
}