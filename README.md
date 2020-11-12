# Books Recommender System

This repository serves as a storage for my solution to Data Scentics interview.
The assignment has two parts:
1. Scientific, where I produce simple ML model for recommendations
2. Engineering, where 
    1. I propose two RS architecture designs 
    2. I use the model from part 1 to build a mini API that takes in the book name and returns the recommendations
  
## Part 1

My solution to part 1 can be found in the [jupyter notebook](https://github.com/smijeva/booksRS/blob/master/recommender.ipynb).
To sum up I:
1. Had a brief look at the data
2. Decided to use simple collaborative approach: knn-clustering
3. Had a look at ratings per book, per users, decided to cut the data with too few ratings
4. Got confused about 0 ratings in the dataset and kind of ignored the issue
5. Built knn-clustering model
6. Developed method for books recommending which uses the developed model and a fuzzy matcher to find the input book (so that their names don't need to be inserted 100% correctly)

Comments on my solution:
- I've spent some rather inadequate amount of time on solving a "KeyError" caused by ISBN keys present in ratings DS and not in books DS
- The performence of the recommender is unmeasured and results seems rather explorative than exploitative (for lord of the rings in didn't get any magic-fantasy books recommendations)
- The solution does not solve potential problem of recommending new books which wouldn't have rating at first (cold-start problem)
- Using fuzzy matcher was convinient choice for the recommender testing

## Part 2.i

In the first architecture, I assume some already well-established running web application with 4 layers (DB, service, API, FE) where is alongside added some recommender service which uses the already exisitng database and maintains the ML model.

![Simple model](https://github.com/smijeva/booksRS/blob/master/simple_recommender.png?raw=true)

The second architecture is a recommender which is web-application(s)-independent and therefore is able to gather data from multiple sources (web applications). The databases for books and ratings are also independent, as there might be need to optimize them differently (e.g. ratings database would require more frequent write operations). The recommender provides two types of API, one which is stable and second one which allows RS's A/B testing. 

![Complex model](https://github.com/smijeva/booksRS/blob/master/complex_recommender.png?raw=true)

## Part 2.ii

The core of this part is file [server.py](https://github.com/smijeva/booksRS/blob/master/server.py) where is just code necessary to run the model copied from the Part 1 and a very simple Flask server.
The solution is also dockerized.

The results is a very simple HTTP server accepting requests on two endpoints:
- `/` root endpoint for "is alive" request
- `/recommend` endpoint which accepts `book`'s name as a form parameter and returns object containing the title of the found book and recommendations for this book

### How to run

**Prerequisites:**
- clone of this repo
- [book reviews datasets](https://www.kaggle.com/ruchi798/bookcrossing-dataset?select=Book+reviews) downloaded and unziped in the same directory as the cloned repo 
- installed docker

**How to build and start the server:**
```shell
$ docker build -t booksRS .
$ docker run -d -p 5000:5000 booksRS
```
**Example of running server usage:**

Server request:
```shell
$ curl -X POST \
  http://localhost:5000/recommend \
  -H 'content-type: multipart/form-data' \
  -F 'book=Great gatsby'
```

Server response:
```json
{
    "book_title": "The Great Gatsby",
    "match_with_query": 86,
    "recommendations": [
        "The Fall of the House of Usher and Other Writings: Poems, Tales, Essays and Reviews (Penguin Classics)",
        "The Complete Prophecies of Nostradamus",
        "The Woodchipper Murder",
        "Eyes Of Fire",
        "The seekers (His The American bicentennial series ; v. 3)",
        "Furies",
        "Blood Roses",
        "Good and Dead (Penguin Crime Monthly)",
        "Mortal Stakes",
        "Rebels"
    ]
}
```

