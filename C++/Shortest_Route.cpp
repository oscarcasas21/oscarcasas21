// Authors: Santiago Quintero & Oscar Casas 
// Uniqnames: squin & casaso
// Date: 12/05/2017
// Lab section: 15 & 12
// Project #6
// filename: planRoute.cpp
// Program Description: 
// This C++ program reads in the locations and names of planets, processes their 
// respective locations and determines the most shortest route for the gLyft driver 
// to reach the customers' planets.

// Libraries included below: #include <iostream> #include <string>
#include <fstream> 
#include <vector>
#include <algorithm> 
#include <cctype> 
#include <functional> 
#include <iostream>
#include <string>
#include <cmath>

using namespace std;





// Planet Struct
// Creating a struct to store planetary data such as the name, the ID of the 
// planet, the location, etc. This will nicely store vectors and make it easier to 
// call the vectors later on.
struct planet{

	int IDName;	
	char symbol;
	double row;
	double col;
	string Name;
	int ID;	
	string visit;

	};





// readFiles
// Reading in the input files
void readFiles(istream & FileLoc, int & StartX, int & StartY, int & FinX, int & FinY, int & NumR, int & NumC){

	FileLoc >> NumR;
	FileLoc >> NumC;
	FileLoc >> StartX;
	FileLoc >> StartY;
	FileLoc >> FinX;
	FileLoc >> FinY;

}





// Reading Location Function
// Get the filenames by prompting the user for the names of the two input files. 
// Do this by using cin and cout.
void locationReadingFunction(istream & FileLoc, planet & singlePlanet, vector<planet> &Planets, istream &Names){

   	while(FileLoc >> singlePlanet.row >> singlePlanet.col >> singlePlanet.symbol >> singlePlanet.IDName) {
   			Planets.push_back(singlePlanet);
   	}
		
		for(int i = 0; i< Planets.size(); ++i){

			int NewID = 0;
			string NewName;
			Names >> NewID;
			Names >> NewName;

		for(int i = 0; i< Planets.size(); ++i){
		
			if(Planets.at(i).IDName == NewID){
			Planets.at(i).Name = NewName;
			}
		}
		}	
}
 	




// Correct Errors
// Replacing all underscores with spaces		
void correct_(vector<planet> &Planets){
 
	string TempName;
	int index;
		for(int i=0; Planets.size() > i; ++i){

			TempName = Planets.at(i).Name;			

			while(TempName.find("_") != TempName.npos){
			index = TempName.find("_");
			TempName = TempName.replace(index,1," ");
				}

			Planets.at(i).Name = TempName;
			}
		}





// Correct Errors
// Find and erase occurrences of "XX"
void correctXX(vector<planet> &Planets){ 
 
	string TempName;
	int index;
		for(int i=0; Planets.size() > i; ++i){			
		
			TempName = Planets.at(i).Name;
			while(TempName.find("XX") != TempName.npos){
			index = TempName.find("XX");
			TempName = TempName.replace(index,2,"");

				}
			
		Planets.at(i).Name = TempName;
		}
	}

		



// Calculate Distance
// Create the distance formula [sqrt((currentx - potentialx)^2 + (currenty - 
// potentialy)^2)] for later use to determine the optimal route.
double calcDistance(double x1, double x2, double y1, double y2){
	
	double distance;
	distance = pow(pow((x1 - x2),2.0) + pow((y1 - y2),2.0),.5);
	return distance;

}





// Grid Function
// First makes a grid of characters to represent the map of the locations that are 
// within the gLyft driver's range. Then mark with an S the starting point in the 
// grid and mark with an E the ending point. And for every other planet, place 
// their corresponding symbol in the grid.
void GridFunction(int rows, int cols, vector<planet> Planets, int StartX, int StartY, int FinX, int FinY, ostream & Fileout){

	for(int i = 0; i < rows; ++i){
		for(int j = 0; j< cols; ++j){
			bool check = true;
	//Starting point
			if(StartX -1  == i & StartY -1 == j){
				Fileout<<"S";
				check = false;
			}
	//Ending point
			if(FinX -1 == i & FinY -1 == j){
				Fileout<<"E";
				check = false;
			}
	//Symbols for planets
		for(int k = 0; k< Planets.size(); ++k){
			if(Planets.at(k).row -1 == i && Planets.at(k).col -1 == j){
				check = false;
				Fileout <<Planets.at(k).symbol;
		}		
	}
	//Anything else without a symbol, assign a period (".")
		if(check == true){
		Fileout<<".";
	}	
}
Fileout << endl;
}	
}			





// Route Determining Function
// We need to first implement the nearest neighbor algorithm 
int routeDeterminingFunction(vector<planet> &Planets, int NewX, int NewY, planet singlePlanet)
{
	
	double min = 9999.0;	
	int index = -1;	
	
		for(int i = 0; i < Planets.size(); ++i){

			double TempDistance= 0.0;	
			double tempX = Planets.at(i).row;
			double tempY = Planets.at(i).col;
	
// calls the distance function and sets the index if it is the smallest distance
		TempDistance = calcDistance(NewX, tempX, NewY, tempY);
		
		if(TempDistance < min){
		min = TempDistance;
		index = i;
		}
// if two distance are the same the idex should favor the planet with the lower 
// Planet ID
		
		if(TempDistance == min && Planets.at(i).IDName < Planets.at(index).IDName){
		
		index = i;
		
		}
		}
		
		return index;
	}
		




// file summarizing the route. 

int main(){
int Row;
int Col;
int StartY;
int StartX;
int FinX;
int FinY;
int row;
int col;
char symbol;
int planetID;
int NumR = 0;
int NumC = 0;

planet singlePlanet;
string LocationFile;
string NameFile;

vector<planet> Planets;
vector<char> Grid;

// Input file containing <location_file>.txt 
// This file contains the location coordinates that the gLyft driver will need to 
// travel to.
	cout << "Enter​ ​Locations​ ​Filename:​ ​";
	cin >> LocationFile;
	ifstream FileLoc(LocationFile.c_str());
	
// Input file containing <name_file>.txt 
// This file contains the location's names that the gLyft driver will need to 
// travel to.	
	cout << "Enter​ ​Names​ ​Filename:​ ";
	cin >> NameFile;
	ifstream Names(NameFile.c_str());

// Open Files
// This will be the file containing the names of the planets with their ID numbers.
// If the file cannot be opened, output an error message.
	if(!FileLoc.is_open() || !Names.is_open()){ //if does not open end    
        cout << "Input file could not be opened." << endl; // message
   		} 
   	// Fileout the file name journey.txt
   	ofstream Fileout("journey.txt");
   	readFiles(FileLoc, StartX, StartY, FinX, FinY, NumR, NumC);
   	
	// Read in the locations
   	locationReadingFunction(FileLoc, singlePlanet, Planets, Names);
	
	// correct the errors
	correct_(Planets);
	correctXX(Planets);
	
	// Make the Grid
	GridFunction(NumR, NumC, Planets, StartX, StartY, FinX, FinY, Fileout);
	vector<planet> Temporary;


	for(int i = 0; i < Planets.size(); ++i){
	
	// Output error message if the planet is out of range.
		if(!((Planets.at(i).row > 0 && Planets.at(i).row <= NumR)&&(Planets.at(i).col  > 0 && Planets.at(i).col <= NumC))){


			cout<< Planets.at(i).IDName <<" out of range - ignoring"<<endl;
	
		} 
	// Push back elements into the temp vector that are within the grid
		else {
			Temporary.push_back(Planets.at(i));
		}

	}
	

// Put the data from the temp vector into the real vector
Planets = Temporary;


// Start the output information
Fileout<<"Start at " << StartX<<" "<< StartY <<endl;


//Initial current x and y cordinates
Row = StartX;
Col = StartY;


// Loop that will run the nearest neighbor algorithim until there are no planets left
while(Planets.size()>0){
	int Place = routeDeterminingFunction(Planets, Row, Col, singlePlanet);
	Row = Planets.at(Place).row;
	Col = Planets.at(Place).col;

	Fileout<<"Go to "<< Planets.at(Place).Name<< " at "<< Planets.at(Place).row<<" " << Planets.at(Place).col<< endl;

	Planets.erase(Planets.begin() + Place);
}


// Final output
Fileout<<"End at "<< FinX<<" "<< FinY<< endl;
}