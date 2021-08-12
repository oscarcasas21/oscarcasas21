CREATE TABLE `Uber_SQL_EX2`.`Exercise2_1` AS
SELECT 
		'',
        driver_id, 
        lap, 
        lap_time_seconds,
        lap_time_seconds - LAG(lap_time_seconds) OVER 
        (PARTITION BY driver_id ORDER BY lap ) AS difference 
        
FROM Results

ORDER BY driver_id, lap;


CREATE TABLE Uber_SQL_EX2.Exercise2_2 AS
SELECT min(ID) AS ID, lap, avg(difference) AS avg_difference
FROM Exercise2_1
GROUP BY lap
ORDER BY lap;

CREATE TABLE Exercise2_3 AS
SELECT round(min(Race_Results.ID)/3,0) AS ID ,driver_id, round(sum(avg_lap_time_seconds) / sum(lap_length_meters),6) AS avg_meters_second
FROM Race_Results
LEFT JOIN circuit_info
ON Race_Results.circuit = circuit_info.circuit
GROUP BY Race_Results.driver_id;


SELECT *
FROM Exercise2_1;

DROP TABLE Exercise2_3;