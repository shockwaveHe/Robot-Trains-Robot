Commonly Asked Questions
========================

MuJoCo
------

**Q: How do I set up headless rendering for MuJoCo?**

A: Add ``export MUJOCO_GL="egl"`` to your ``.bashrc`` if headless
rendering is desired. Then ``source ~/.bashrc``.

**Q: How do I run MuJoCo on MacOS?**

A: Run MuJoCo-related scripts with ``mjpython`` instead of ``python``.

**Q: What should I do if there is an error like this when activating the conda environment?**

    .. code:: bash

            active environment : None
                    shell level : 4
            user config file : /lustre/home/jiayou.zhang/.condarc
        populated config files : 
                conda version : 23.11.0
            conda-build version : not installed
                python version : 3.11.5.final.0
                        solver : libmamba (default)
            virtual packages : __archspec=1=zen3
                                __conda=23.11.0=0
                                __glibc=2.35=0
                                __linux=5.15.0=0
                                __unix=0=0
            base environment : /lustre/home/jiayou.zhang/miniconda3  (writable)
            conda av data dir : /lustre/home/jiayou.zhang/miniconda3/etc/conda
        conda av metadata url : None
                channel URLs : https://repo.anaconda.com/pkgs/main/linux-64
                                https://repo.anaconda.com/pkgs/main/noarch
                                https://repo.anaconda.com/pkgs/r/linux-64
                                https://repo.anaconda.com/pkgs/r/noarch
                package cache : /lustre/home/jiayou.zhang/miniconda3/pkgs
                                /lustre/home/jiayou.zhang/.conda/pkgs
            envs directories : /lustre/home/jiayou.zhang/miniconda3/envs
                                /lustre/home/jiayou.zhang/.conda/envs
                    platform : linux-64
                    user-agent : conda/23.11.0 requests/2.31.0 CPython/3.11.5 Linux/5.15.0-60-generic ubuntu/22.04.2 glibc/2.35 solver/libmamba conda-libmamba-solver/23.12.0 libmambapy/1.5.3
                        UID:GID : 923002912:923000513
                    netrc file : None
                offline mode : False


        An unexpected error has occurred. Conda has prepared the above report.
        If you suspect this error is being caused by a malfunctioning plugin,
        consider using the --no-plugins option to turn off plugins.

        Example: conda --no-plugins install <package>

        Alternatively, you can set the CONDA_NO_PLUGINS environment variable on
        the command line to run the command without plugins enabled.

        Example: CONDA_NO_PLUGINS=true conda install <package>


        Timeout reached. No report sent.


A: Consider `this solution <https://github.com/conda/conda/issues/13451#issuecomment-1897540968>`__.


.. note::

    If you ever find yourself facing a bug that occurs sporadically and goes away by itself. Double check all the wirings. It could be a loose cable somewhere

**Q: We once found a it can not identify any motor on dynamixel wizard, but if we only plug in the individual chains, it works no problem. All the motor and u2d2 are good. And the problem occasionally disappears and all motors comes back.**


A: We found a loose cable near the ankle. It's just loose enough to disconnect on motion and come back to contact after. It is suspected to dump stuff into the data line when it reboots and mess with the communication of the rest of the motors. 


**Q: If there is a `yourdfpy` import error like this.**

A: ``pip install lxml==4.9.4``