from setuptools import setup, find_packages

setup(name = 'gym_CTgraph',
      version = '0.1',
      install_requires = ['gym'],  # And any other dependencies foo needs
      packages = find_packages(),
      author = "Andrea Soltoggio, Pawel Ladosz, Yang Hu, Eseoghene Iwhiwhu",
      author_email = "a.soltoggio@lboro.ac.uk",
      description = "This is the implementation for the configurabe tree graph (CT-graph)",
      license = "Copyright (C) 2019  Andrea Soltoggio. This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.",
      keywords = "Deep reinforcement learning, dynamic rewards, continual learning, adaptation, partially observable Markov decision problems, POMDP",
      url = "https://github.com/soltoggio/CT-graph",
      project_urls = {
        "Bug Tracker": "",
        "Documentation": "",
        "Source Code": "https://github.com/soltoggio/CT-graph",
    }
)
