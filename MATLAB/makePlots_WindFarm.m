%Oscar Casas "casaso",001, Santiago Quintero "squin", 001, 24/10/17
function [  ] = makePlots( filenameWind, filenameWave, filenameBuoy, windSpeedMin, windSpeedMax, waveHeightMax )

% Function to complete Task 2. Creates a figure with multiple plots that 
% summarizes the environmental conditions for a wind farm.  Saves figure as 
% a .png file.
%
%   parameters: 
%          filenameWind: a string that names the file containing the 
%                        global-model-based average wind speed 
%                        (i.e. 'windSpeedTestCase.csv')
%          filenameWave: a string that names the file containing the 
%                        global-model-based average global wave heights 
%                        (i.e. 'waveHeightTestCase.csv')
%          filenameBuoy: a string that names the file containing the time 
%                        series of wave heights measured by the buoy          
%                        (i.e. 'buoyTestCase.csv')
%          windSpeedMin: for constraint 1 -- minimum wind speed (m/s)
%          windSpeedMax: for constraint 1 -- maximum wind speed (m/s)
%         waveHeightMax: for constraint 2 -- maximum wave height (m)
%
%   return values: none
%
%   notes:
%       Feel free to use different variable names than these if it makes 
%       your code more readable to you.  These are internal to your 
%       function, so it doesn't matter what you call them.

%% Load the data
wind = csvread(filenameWind);
wave = csvread(filenameWave);
Buoy = csvread(filenameBuoy, 5,0);

buoyLocation = csvread(filenameBuoy,1,1,[1 1 1 2]);
Buoylat = buoyLocation(1,1);
Buoylon = buoyLocation(1,2);
Time = Buoy(:,1);
buoyHeight = Buoy(:,2);
% Get lat/lon data
lat = csvread('lat.csv');
lon = csvread('lon.csv');

% Read in the rest of the data you need...


%% Figure Setup

% Set up the figure properties so it will be the correct size
Fig1 = figure(1);
Fig1.Units = 'character';
%Fig1.Position = [0, 0, 120, 60]; % uncomment this line if you use a MAC
Fig1.Position = [0, 0, 150, 60]; % uncomment this line if you use WINDOWS
                                  % if you use Linux, fiddle with the
                                  % width (the 3rd number) until the figure
                                  % looks about right
Fig1.PaperPositionMode = 'auto';

%% Make Plots


% Make the plots...
windSpeedTestCase = csvread('windSpeedTestCase.csv');




%plot 1st graph
graph = subplot(3,2,1);
%meshgrid the data
[X,Y] = meshgrid(lon,lat);

%Countour and color in the graph
contourf(X,Y,wind,'LineStyle','none')
colormap(graph,'parula')
colorbar('eastoutside')

%axis labels and rest
title('Average Wind Speed (m/s) ACross Planet')
xlabel('longitude (deg)')
ylabel('latitude (deg)')
axis([0 300 -100 100])






%plot 2nd graph
graphDos = subplot(3,2,2);
%meshgrid the data
[X Y] = meshgrid(lon,lat);

contourf(X,Y,wave,'LineStyle','none')
colormap(graphDos,'parula')
colorbar('eastoutside')


%axis labels and rest
title('Average Wind Speed (m/s) ACross Planet')
xlabel('longitude (deg)')
ylabel('latitude (deg)')
axis([0 300 -100 100])




%plot 3rd graph
graphTres = subplot(3,2,3);
c1 = (windSpeedMin < wind(Buoylat,Buoylon) & wind(Buoylat,Buoylon) < windSpeedMax);
c2 = wave(Buoylat,Buoylon) < waveHeightMax;

constraints = c1 & c2;

%meshgrid the data
[X Y] = meshgrid(lon,lat);
contourf(X,Y,wind,'LineStyle','none')
colormap(graphTres,flipud(gray))
colorbar('eastoutside')

%axis labels and rest
axis([0 300 -100 100])
title('Potential Farm Wind Locations')
xlabel('longitude (deg)')
ylabel('latitude (deg)')

hold on;
%coordinates of the ideal location for a wind farm
longitude = lon(Buoylon);
latitude = lat(Buoylat);

plot(longitude,latitude,'rs','MarkerSize',12);
hold off





%plot 4th graph
subplot(3,2,4)
histogram(buoyHeight)  

%axis labels and rest
axis ([0 10 0 100]);
xlabel('wave height (m)');
ylabel('number of occurrences');
title('Wave Heights at Buoy Location');

%turn on the grid
grid on;
 


%plot 5th graph
subplot(3,1,3)
plot(Time, buoyHeight, '-b');

%axis labels and rest
axis([0 4500 0 10])
xlabel('time (hours)');
ylabel('wave height (m)');
title('Wave Height Comparison: Global to Local (by Santiago Quintero)');

%turn on the grid
grid on

%From analyzewindfarm
buoyLocation = csvread(filenameBuoy,1,1,[1 1 1 2]);
buoyLat = buoyLocation(1,1);
buoyLon = buoyLocation(1,2);

%Get vector with locations to plot the global average
coordinates = wave(buoyLat,buoyLon);
vectorWaveHeights = ones(1,length(buoyHeight));
vectorLocation = coordinates .* vectorWaveHeights;
hold on
plot(Time, vectorLocation, '-r');
hold off
%legend always finish last
legend('Buoy-measured', 'Global Average', 'Location', 'northeast');

%Save as a png
print('environmentalSummary.png','-dpng');

end

