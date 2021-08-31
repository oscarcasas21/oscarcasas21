// Author: Oscar Casas
// Uniqname: casaso
// Date: 11/30/2017
// Lab Section: 012 & 015
// Project #: 5
// Filename: evaluateReviews.cpp
// Program Description: 
// This program is the main function that evaluates reviews and gives a report 
// evaluating the truthfulness or how deceitful a review is

// Add any #includes for C++ libraries here.
// We have already included iostream as an example.

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cctype>
#include <functional>
#include "reviews.h"

using namespace std;

const double SCORE_LIMIT_TRUTHFUL = 3; // create limits
const double SCORE_LIMIT_DECEPTIVE = -3; // create limits

int main() {

ifstream KeywordWeights("keywordWeights.txt"); // read in keyword weights

    if(!KeywordWeights.is_open()){ //if does not open end
    
        cout << "Error: keywordweights.txt could not be opened." << endl; // message

    return 1;
   	}

vector<string> keywords; // call variables
vector<double> weights; // call variables

readKeywordWeights(KeywordWeights, keywords, weights); // call function to get vectos for weight and keywords.
 
	ofstream Fileout("report.txt"); // set output file
	
	Fileout << "review score category" << endl; // title

int NumberDeceptive = 0;// call variables
int NumberTrue = 0;// call variables
int NumberUncategorized = 0;// call variables
int BigTrueReview = 0;// call variables
double BigTrueWeight = 0.0;// call variables
double BigDecepWeight = 0.0;// call variables
int BigRecepReview = 0;// call variables

	for(int i = 0; i <= 100; ++i){ // limits
	
	string Reviewfilename = makeReviewFilename(i); // use function given
	ifstream review(Reviewfilename); // open file 

	if(!review.is_open()){ // when it does not open show this
	Fileout << endl;
	Fileout << "number of reviews: " << i << endl;
	Fileout << "number of truthful reviews: " << NumberTrue << endl;
	Fileout << "number of deceptive reviews: " << NumberDeceptive << endl;
	Fileout << "number of uncategorized reviews: " << NumberUncategorized << endl;

	Fileout << endl;
	Fileout << "Most truthful Review: " << BigTrueReview << endl;
	Fileout << "Most deceptive Review: " << BigRecepReview << endl;
	return 1;
	}

	vector<string> reviewWords; // call variables

	readReview(review, reviewWords); // call fuction readReaview to create vector out of files.


	double totalscore = reviewScore(reviewWords, keywords, weights); // call variables value from the function

	string label; // call variables



if (totalscore > SCORE_LIMIT_TRUTHFUL) { //Finding true values
	label = "truthful";
	NumberTrue += 1;
	
}
if (totalscore < SCORE_LIMIT_DECEPTIVE) { // Finding Deceptive Valeues
		
	label = "deceptive";
	NumberDeceptive += 1;
			
	}

if (totalscore <= SCORE_LIMIT_TRUTHFUL && totalscore >= SCORE_LIMIT_DECEPTIVE){ // Finding uncategorized values
		label = "uncategorized";
		NumberUncategorized += 1;
		

	}

Fileout << i << " " << totalscore << " " << label << endl; // output when the if statement has passed

if (totalscore > BigTrueWeight) { // find most truthful review
		BigTrueWeight = totalscore;
		BigTrueReview = i;
		}

if (totalscore < BigDecepWeight) { // find most deceptive review
		BigDecepWeight = totalscore;
			BigRecepReview = i;
		}

}
KeywordWeights.close(); //close input file
Fileout.close(); // close output file
}
