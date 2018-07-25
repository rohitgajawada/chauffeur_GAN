# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# CORL experiment set.

from __future__ import print_function


from carla.driving_benchmark.experiment import Experiment
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.driving_benchmark.experiment_suites.experiment_suite import ExperimentSuite


#TODO: maybe add aditional tasks ( NO dynamic obstacles for instace !)

class TestSuite(ExperimentSuite):

    def __init__(self):
        super(TestSuite, self).__init__('Town02')

    @property
    def train_weathers(self):
        return [12]
    @property
    def test_weathers(self):
        return [12]


    def _poses(self):


        return [[[3, 23], [0,10],[7, 2], [3, 17], [6, 19], [4, 11]]]



    def build_experiments(self):
        """
        Creates the whole set of experiment objects,
        The experiments created depend on the selected Town.


        """

        # We set the camera
        # This single RGB camera is used on every experiment

        camera = Camera('rgb')
        camera.set(FOV=90)
        camera.set_image_size(671, 375)
        camera.set_position(0.2, 0.0, 0.85)
        camera.set_rotation(-3.0, 0, 0)

        poses_tasks = self._poses()
        vehicles_tasks = [0, 0, 0, 15]
        pedestrians_tasks = [0, 0, 0, 50]



        experiments_vector = []

        for weather in self.weathers:

            for iteration in range(len(poses_tasks)):
                poses = poses_tasks[iteration]
                vehicles = vehicles_tasks[iteration]
                pedestrians = pedestrians_tasks[iteration]

                conditions = CarlaSettings()
                conditions.set(
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=vehicles,
                    NumberOfPedestrians=pedestrians,
                    WeatherId=weather,
                    QualityLevel='Epic'
                )
                # Add all the cameras that were set for this experiments

                conditions.add_sensor(camera)

                experiment = Experiment()
                experiment.set(
                    Conditions=conditions,
                    Poses=poses,
                    Task=iteration,
                    Repetitions=1
                )
                experiments_vector.append(experiment)

        return experiments_vector
