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
