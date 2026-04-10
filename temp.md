RyanW@RWatachLaptop MINGW64 ~/robo-greeter/robo-greeter (main)
$ git pull
remote: Enumerating objects: 4, done.
remote: Counting objects: 100% (4/4), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 3 (delta 1), reused 0 (delta 0), pack-reused 0 (from 0)
Unpacking objects: 100% (3/3), 1.47 KiB | 125.00 KiB/s, done.
From https://github.com/ryanwatach/robo-greeter
   71de467..13cdc9a  main       -> origin/main
Updating 8f8e6d5..13cdc9a
error: Your local changes to the following files would be overwritten by merge:
        main.py
Please commit your changes or stash them before you merge.
Aborting

RyanW@RWatachLaptop MINGW64 ~/robo-greeter/robo-greeter (main)
$ git stash
error: lstat("data/faces/Open_So_Everything_Start_Only_And_Then_Try_To_Kill_A_Kill_Wall_You_Have_Multiple_The_Halves_Open_So_By_Multiple_Multiple_Them_Early_You_See_How_The_Restarting_Add_In_A_Follow-up_Hey_Well_Sing_You_In_Okay_All_In_Some_All_In_Some_1774450511.jpg"): Filename too long
fatal: Unable to process path data/faces/Open_So_Everything_Start_Only_And_Then_Try_To_Kill_A_Kill_Wall_You_Have_Multiple_The_Halves_Open_So_By_Multiple_Multiple_Them_Early_You_See_How_The_Restarting_Add_In_A_Follow-up_Hey_Well_Sing_You_In_Okay_All_In_Some_All_In_Some_1774450511.jpg
Cannot save the current worktree state

RyanW@RWatachLaptop MINGW64 ~/robo-greeter/robo-greeter (main)
$ git pull
Updating 8f8e6d5..13cdc9a
error: Your local changes to the following files would be overwritten by merge:
        main.py
Please commit your changes or stash them before you merge.
Aborting
y m
RyanW@RWatachLaptop MINGW64 ~/robo-greeter/robo-greeter (main)
$ py main.py
C:\Users\RyanW\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\face_recognition_models\__init__.py:7: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import resource_filename
Traceback (most recent call last):
  File "C:\Users\RyanW\robo-greeter\robo-greeter\main.py", line 630, in <module>
    main()
    ~~~~^^
  File "C:\Users\RyanW\robo-greeter\robo-greeter\main.py", line 405, in main
    with open(LOCKFILE, "w") as _f:
         ~~~~^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/robo-greeter.pid'

RyanW@RWatachLaptop MINGW64 ~/robo-greeter/robo-greeter (main)
$
