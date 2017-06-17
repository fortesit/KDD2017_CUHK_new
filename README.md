https://github.com/kin-cs/Kaggle_Amazon_2017/invitations

# KDD2017_CUHK
KDD 2017 team from CUHK

# Preprocessed Data Description
https://github.com/studdykid/KDD2017_CUHK/wiki/About-joined-csv-table


# Data Feature:
## Basic Info
![Info1](https://img.alicdn.com/tps/TB1gP0QPVXXXXaQXFXXXXXXXXXX-809-267.png)


## Chinese Holidays:
  There were two Chinese holidays during the time of the dataset.

  1. Sep.15 to Sep. 17, 2016 was the Moon Festival holiday, while Sep. 18 (Sunday) was a working day.

  2. Oct. 1 to Oct. 7, 2016 was the Chinese national holiday week, while Oct. 8 - 9 (Saturday & Sunday) were a working days.

  During Chinese national holiday week, highway tolls were waived for passenger vehicles.

### Road Network id
![Road_Network_id](https://raw.githubusercontent.com/kin-cs/KDD2017_CUHK_new/master/etc/Data%20Visualization/phase1_road_network_id.png)

### Road Network lanes
![Road_Network_lanes](https://raw.githubusercontent.com/kin-cs/KDD2017_CUHK_new/master/etc/Data%20Visualization/phase1_road_network_lanes.png)

### Road Network length
![Road_Network_length](https://raw.githubusercontent.com/kin-cs/KDD2017_CUHK_new/master/etc/Data%20Visualization/phase1_road_network_length.png)


## Checkpoints (Volume)
5 Checkpoint: 1-0 , 1-1, 2-0, 3-0, 3-1
    2nd index{ 0: entry, 1: exit }

## Lanes (Traj)
6 Paths: A-2, A-3, B-1, B-3, C-1, C-3

# Datasets:
```
[Phase 1]
- Trajectories [2016, 7,19] - [2016,10,17]
- Volume       [2016, 9,19] - [2016,10,17]
- Weather      [2016, 7, 1] - [2016,10,17]

[Test Phase 1]
- Weather      [2016,10,18] - [2016,10,24]

[Phase 2]
- Trajectories [2016,10,18] - [2016,10,24]
- Volume       [2016,10,18] - [2016,10,24]
- Weather      [2016,10,25] - [2016,10,31]

[Test (Final)]
- Trajectories [2016,10,25] - [2016,10,31] (6:00-7:59 & 15:00-16:59 only)
- Volume       [2016,10,25] - [2016,10,31] (6:00-7:59 & 15:00-16:59 only)


=== In conclusion, totally we have... ===
[Training Dataset]
- Trajectories [2016, 7,19] - [2016,10,24]
- Volume       [2016, 9,19] - [2016,10,24]
- Weather      [2016, 7, 1] - [2016,10,24]

[Test (Final)]
- Trajectories [2016,10,25] - [2016,10,31] (6:00-7:59 & 15:00-16:59 only)
- Volume       [2016,10,25] - [2016,10,31] (6:00-7:59 & 15:00-16:59 only)
- Weather      [2016,10,25] - [2016,10,31] (whole day)

(All data above contain entire 24 hrs, except Final Test data.)
```
## Features:
1. Volume (total) (20m)
-
2. ETC (20m)
3. Motorcycle (20m)
4. Cargocar (20m)
5. Privatecar (20m)
6. Unknowncar (20m)
-
7. routetime median (20m)
-
8. pressure (3 hrs)
9. sea_pressure (3 hrs)
10. wind_direction (3 hrs)
11. wind_speed (3 hrs)
12. temperature (3 hrs)
13. rel_humidity (3 hrs)
14. precipitation (3 hrs)

--
15. date
16. hour
17. dayofweek
18. is_holiday


## TODO:

1. Create the Input Tensor (3D)

## Done:
1. Investigate Backfill for FillNA: Solved by interpolate (by Panda function)
2. Moving avg for 3hr's features (interpolate)
