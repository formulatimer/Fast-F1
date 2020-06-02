from fastf1 import core, api
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt
import pickle
import IPython
from multiprocessing import Process, Manager
import time


"""
Distinction between "Time" and "Date":

Time:   A time stamp counting up from the start of the session.
        Might sometimes be called session time for sake of clarity.
        Format: HH:MM:SS.000
        
Date:   The actual date and time at which something happened.
        Timezone is UTC I think.
        Format: YYYY-MM-DD HH:MM:SS.000
        
The terms time and date will be used consistently with this meaning.
"""


class TrackPoint:
    """Simple point class.

    A point has an x and y coordinate and an optional date.
    A function for calculating the square of the distance to another point is also provided.

    For convenience reasons point.x and point.y can also be accessed as point['x'] and point['y'] (get only).
    Only use this implementation when necessary because it lacks clarity. It's not quite obvious then that poitn is a
    separate class (could be a dictionary for example)
    """
    def __init__(self, x, y, date=None):
        """
        :param x: x coordinate
        :type x: int or float
        :param y: y coordinate
        :type y: int or float
        :param date: optional: A pandas datetime (compatible) object
        """
        self.x = x
        self.y = y
        self.date = date

    def __getitem__(self, key):
        if key == 'x':
            return self.x
        elif key == 'y':
            return self.y
        else:
            raise KeyError

    def get_sqr_dist(self, other):
        """Calculate the square of the distance to another point.

        :param other: Another point
        :type other: TrackPoint
        :return: distance^2
        """
        dist = abs(other.x - self.x) + abs(other.y - self.y)
        return dist


class Track:
    # TODO reorder points when start finish line position is known
    """Track position related data processing.

    The unit (if any) of F1's coordinate system is unknown to me. Approx.: value / 3,61 = value in meters

    Although there are more than one hundred thousand points of position data per session, the number of
    unique points on track is limited. Typically there is about one unique point per meter of track length.
    This does not mean that the points have a fixed distance though. In slow corners they are closer together than
    on straights. A typical track has between 5000 and 7000 unique points.
    When generating the track map, all duplicate points are removed from the raw data so that only unique points are left.
    Then those points are sorted into the correct order.
    Not all unique points of a given track are necessarily present in each session. There is simply a chance that position
    data from no car is ever sent from a point. In this case we can't know that this point exist.
    This is not a problem, but the following needs to be kept in mind:

    The track class and its track map are only a valid representation of the data they were calculated from.
    E.g. do not use a track map for race data when it was calculated from qualifying data.

    Sharing a track map between multiple sessions may be possible if the all points from all of these
    sessions where joined before and the track map was therefore calculated from both sessions at the same time.
    Although this may be possible, it is neither tested, nor intended, recommended or implemented (yet). If consistency
    of position data between sessions can be validated, this might be a way of getting more data and thereby increased accuracy of
    some statistically computed values.
    """

    def __init__(self, pos_frame):
        """Create a new track map object.

        :param pos_frame: Dictionary containing a pandas.DataFrame with position data per car (as returned by fastf1.api.position)
        :type pos_frame: dict
        """

        self._pos_data = pos_frame

        self.unsorted_points = list()
        self.sorted_points = list()
        self.excluded_points = list()

        self.sorted_x = list()  # list of sorted coordinates for easy plotting and lazy coordinate validation
        self.sorted_y = list()

        self.distances = list()
        self.distances_normalized = list()

        self.track = None

        self._next_point = None

        self._vis_freq = 0
        self._vis_counter = 0
        self._fig = None

        # extract point from position data frame
        self._unsorted_points_from_pos_data()

    def _unsorted_points_from_pos_data(self):
        """Extract all unique track points from the position data."""
        # get all driver numbers
        drivers = list(self._pos_data.keys())

        # create a combined data frame with all column names but without data; use the data of the first driver to get the column names
        combined = pd.DataFrame(columns=[*self._pos_data[drivers[0]].columns])

        # add the data of all drivers
        for n in drivers:
            combined = combined.append(self._pos_data[n])

        # filter out data points where the car is not on track
        is_on_track = combined['Status'] == 'OnTrack'
        combined = combined[is_on_track]

        # filter out anything but X and Y coordinates and drop duplicate values
        no_dupl_combined = combined.filter(items=('X', 'Y')).drop_duplicates()

        # create a point object for each point
        for index, data in no_dupl_combined.iterrows():
            self.unsorted_points.append(TrackPoint(data['X'], data['Y']))

    def _init_viusualization(self):
        """Initiate the plot for visualizing the progress of sorting the track points."""
        self._vis_counter = 0
        plt.ion()
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        self._ax.axis('equal')
        self._line2, = self._ax.plot((), (), 'r-')
        self._line1, = self._ax.plot((), (), 'b-')

    def _cleanup_visualization(self):
        """Clean up the sorting visualization plot and 'reset' matplotlib so following plots are not influenced."""
        if self._fig:
            plt.close()
            plt.ioff()
            plt.clf()
            self._fig = None

    def _visualize_sorting_progress(self):
        """Visualize the current progress of sorting of the track points.

        Updates the plot with the current data. The plot is created first if this is
        the first call to this function.
        """
        if not self._vis_freq:
            return  # don't do visualization if _vis_freq is zero

        if not self._fig:
            self._init_viusualization()  # first call, setup the plot

        self._vis_counter += 1

        if self._vis_counter % self._vis_freq == 0:
            # visualize current state
            xvals_sorted = list()
            yvals_sorted = list()
            for point in self.sorted_points:
                xvals_sorted.append(point.x)
                yvals_sorted.append(point.y)

            xvals_unsorted = list()
            yvals_unsorted = list()
            for point in self.unsorted_points:
                xvals_unsorted.append(point.x)
                yvals_unsorted.append(point.y)

            # update plot
            self._line1.set_data(xvals_sorted, yvals_sorted)  # set plot data
            self._line2.set_data(xvals_unsorted, yvals_unsorted)  # set plot data
            self._ax.relim()  # recompute the data limits
            self._ax.autoscale_view()  # automatic axis scaling
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

    def _integrate_distance(self):
        """Integrate distance over all points and save distance from start/finish line for each point."""
        # TODO this is currently not implemented; need start/finish line position and direction. Maybe then save results per point in point object
        self.distances.append(0)  # distance is obviously zero at the starting point

        distance_covered = 0  # distance since first point

        for i in range(1, len(self.sorted_points)):
            # calculate the length of the segment between the last and the current point
            segment_length = sqrt(self.sorted_points[i-1].get_sqr_dist(self.sorted_points[i]))
            distance_covered += segment_length
            self.distances.append(distance_covered)

        for dist in self.distances:
            self.distances_normalized.append(dist / self.distances[-1])

    def _determine_track_direction(self):
        """Check if the track direction is correct and if not reverse the list of sorted points to correct it.

        This is done by getting two arbitrary points from the position (telemetry) data. Then it is checked that the first
        of these two points is also first in the list of sorted points. If not, the list is reversed.
        """
        drivers = list(self._pos_data.keys())
        n = 0
        while True:
            drv = drivers[n]  # select a driver
            on_track = self._pos_data[drv][self._pos_data[drv].Status == "OnTrack"]  # use 2nd lap as a sample lap

            if on_track.empty:
                n += 1  # this driver was never on track; try the next driver
                continue

            try:
                p1 = on_track.iloc[100]  # get two points; doesn't really matter which points are used
                p2 = on_track.iloc[101]
            except IndexError:
                n += 1  # driver wasn't on track very long apparently; try the next driver
                continue

            point1 = self.get_closest_point(TrackPoint(p1.X, p1.Y))  # the resulting point will have the same coordinates but the exact instance
            point2 = self.get_closest_point(TrackPoint(p2.X, p2.Y))  # is required to get its index in the next step

            idx1 = self.sorted_points.index(point1)
            idx2 = self.sorted_points.index(point2)

            if idx1 > idx2 and not (idx1 - idx2) > 0.9 * len(self.sorted_points):
                # first part of this check: The point with the higher index is the one which is later in the lap. This should be the second point.
                #                           If this is not the case, the list needs to be reversed.
                # second part: The exception is, if the list divides the track between these two points. In this case the first point would have
                #               a higher index because it right at the end of the list while the second point is at the beginning. In case that
                #               more than 90% of the list are between these two points this edge case is assumed. The list will not be reversed.
                self.sorted_points.reverse()

            break

    def _sort_points(self):
        """Does the actual sorting of points."""
        # Get the first point as a starting point. Any point could be used as starting point. Later the next closest point is used as next point.
        self._next_point = self.unsorted_points.pop(0)

        while self.unsorted_points:
            self._visualize_sorting_progress()

            # calculate all distances between the next point and all other points
            distances = list()
            for pnt in self.unsorted_points:
                distances.append(self._next_point.get_sqr_dist(pnt))

            # get the next closest point and its index
            min_dst = min(distances)
            index_min = distances.index(min_dst)

            # Check if the closest point is within a reasonable distance. There are some outliers which are very clearly not on track.
            # The limit value was determined experimentally. Usually the distance between to points is approx. 100.
            # (This is the square of the distance. Not the distance itself.)
            # If the _next_point has no other point within a reasonable distance, it is considered an outlier and removed.
            if min_dst > 200:
                self.excluded_points.append(self._next_point)
            else:
                self.sorted_points.append(self._next_point)

            # Get a new _next_point. The new point is the one which was closest to the last one.
            self._next_point = self.unsorted_points.pop(index_min)

        # append the last point if it is not an outlier
        if self._next_point.get_sqr_dist(self.sorted_points[-1]) <= 200:
            self.sorted_points.append(self._next_point)
        else:
            self.excluded_points.append(self._next_point)

        self._cleanup_visualization()

    def generate_track(self, visualization_frequency=0):
        """Generate a track map from the raw points.

        Sorts all points. Then determines the correct direction and starting point (both not implemented yet).
        Finally the lap distance is calculated by integrating over all points (implemented partially, not enabled, depending on previous).
        The distance since start is saved for each point. Additionally, the lap distance is saved normalized to a range of 0 to 1.
        :param visualization_frequency: (optional) specify  after how many calculated points the plot should be updated.
            Set to zero for never (default: never). Plotting is somewhat slow. A visualization frequency greater than 50 is recommended.
        :type visualization_frequency: int
        """
        self._vis_freq = visualization_frequency

        self._sort_points()
        self._determine_track_direction()

        for point in self.sorted_points:
            self.sorted_x.append(point.x)
            self.sorted_y.append(point.y)

        # self._integrate_distance()  # TODO this should not be done before determining track direction and start/finish line position

        # xvals = list()  # TODO rethink this
        # yvals = list()
        # for point in self.sorted_points:
        #     xvals.append(point.x)
        #     yvals.append(point.y)
        #
        # self.track = pd.DataFrame({'X': xvals,
        #                            'Y': yvals,
        #                            'Distance': self.distances,
        #                            'Normalized': self.distances_normalized})

    def lazy_is_track_point(self, x, y):
        """Lazy check for whether two coordinates are the coordinates of a unique track point.

        This function only checks both coordinates independently. But it does not verify that the
        combination of both coordinates is a valid unique track point (therefore "lazy" check).
        This is an intentional measure for saving time.
        """
        if x in self.sorted_x and y in self.sorted_y:
            return True
        return False

    def get_closest_point(self, point):
        """Find the closest unique track point to any given point.

        'point' can be an arbitrary point anywhere. If 'point' is one of the unique track points
        the same point will be returned as no point is closer to it than itself.

        This function assumes that the track is made up of all possible points.
        This assumption is valid within the scope of the data from which the track was calculated.
        See disclaimer for track map class in general

        :param point: A point with arbitrary coordinates
        :type point: TrackPoint
        :return: A single TrackPoint
        """

        distances = list()
        for track_point in self.sorted_points:
            distances.append(track_point.get_sqr_dist(point))

        return self.sorted_points[distances.index(min(distances))]

    def get_points_between(self, point1, point2, short=True, include_ref=True):
        """Returns all unique track points between two points.

        'point1' and 'point2' must be unique track points. The cannot be points with random coordinates.
        If you want to use any given point, call .get_closest_point() first to get the unique track point
        which is closest to your point. You can then pass it as reference point to this function.

        :param point1: First boundary point
        :type point1: TrackPoint
        :param point2: Second boundary point
        :type point2: TrackPoint
        :param short: Whether you want to have the result going the long or the short distance between the boundary points.
        :param include_ref: Whether to include the given boundary points in the returned list of points
        :return: List of TrackPoints
        """

        i1 = self.sorted_points.index(point1)
        i2 = self.sorted_points.index(point2)

        if abs(i1 - i2) < 0.5 * len(self.sorted_points):
            short_is_inner = True
        else:
            short_is_inner = False

        if (short and short_is_inner) or (not short and not short_is_inner):
            # the easy way, simply slice between the two indices
            pnt_range = self.sorted_points[min(i1, i2)+1: max(i1, i2)]
            if include_ref:
                if i1 < i2:
                    pnt_range.insert(0, point1)
                    pnt_range.append(point2)
                else:
                    pnt_range.insert(0, point2)
                    pnt_range.append(point1)

        else:
            # the harder way; we need the first and last part of the list but not the middle
            first = self.sorted_points[:min(i1, i2)]
            second = self.sorted_points[max(i1, i2)+1:]

            # add the reference points correctly
            # also reverse if necessary to keep a consistent returned order. the first reference point should also be first in the result
            if i1 > i2:
                if include_ref:
                    second.insert(0, point1)
                    first.append(point2)
                pnt_range = second + first
            else:
                if include_ref:
                    second.insert(0, point2)
                    first.append(point1)
                pnt_range = second + first
                pnt_range.reverse()

        return pnt_range

    def get_second_coord(self, val, ref_point_1, ref_point_2, from_coord='x'):
        """Calculate the second coordinate if either x or y are known.

        The known coordinate does not need to be the coordinate of a unique track point. The result
        will be interpolated.
        This requires two reference points between which the point your interested in is located.
        This is somewhat unstable. If the range between the two points is to long, there might be
        multiple possible results for your value. In this case this function will fail silently!
        One of the results will be returned.
        The track between the two given points should be approximtely straight for this function to
        work correctly. If the value you're interested in is in a corner, the corner segment between
        ref_point_1 and ref_point_2 should be sufficiently short.
        The reference points need to be unique track points.

        :param val: known x or y coordinate
        :type val: int or float
        :param ref_point_1: First boundary point
        :type ref_point_1: TrackPoint
        :param ref_point_2: Second boundary point
        :type ref_point_2: TrackPoint
        :param from_coord: Specify whether the given value is the x or y coordinate; one of 'x', 'y'
        :type from_coord: str
        :return: TrackPoint
        """
        p_range = self.get_points_between(ref_point_1, ref_point_2)

        # find the closest point in this range; only valid if the range is approximately straight
        # because we're only checking against one coordinate
        distances = list()
        for p in p_range:
            distances.append(abs(p[from_coord] - val))

        min_i = min_index(distances)
        p_a = p_range[min_index(distances)]  # closest point
        # second closest point (with edge cases if closest point is first or last point in list)
        # This works because the points returned by get_points_between() are sorted. The second
        # closest point therefore needs to be the one before or after the closest point.
        if min_i == 0:
            p_b = p_range[1] if distances[1] < distances[-1] else p_range[-1]
        elif min_i == len(distances) - 1:
            p_b = p_range[0] if distances[0] < distances[-2] else p_range[-2]
        else:
            p_b = p_range[min_i+1] if distances[min_i+1] < distances[min_i-1] else p_range[min_i-1]

        # do interpolation
        delta_x = p_b.x - p_a.x
        delta_y = p_b.y - p_a.y

        if from_coord == 'x':
            interp_delta_x = val - p_a.x
            interp_y = p_a.y + delta_y * interp_delta_x / delta_x
            return TrackPoint(val, interp_y)
        else:
            interp_delta_y = val - p_a.y
            interp_x = p_a.x + delta_x * interp_delta_y / delta_y
            return TrackPoint(interp_x, val)

    def get_time_from_pos(self, drv, point, time_range_start, time_range_end):
        """Calculate the time at which a driver was at a specific coordinate.

        The point can be any point. It does not need to be a unique track point.
        A time range needs to be specified because of course a driver passes all parts of the track
        once every lap (surprise there...). The specified time range should therefore be no longer
        than one lap so that there are not multiple possible solutions.
        But shorter is faster in terms of calculating the result. So keep it as short as possible.
        :param drv: Number of the driver as a string
        :type drv: str
        :param point: The point you're interested in
        :type point: TrackPoint
        :param time_range_start: A pandas.Timestamp compatible date
        :param time_range_end: A pandas.Timestamp compatible date
        :return: pandas.Timestamp or None
        """
        drv_pos = self._pos_data[drv]  # get DataFrame for driver

        # calculate closest point in DataFrame (a track map contains all points from the DataFrame)
        closest_track_pnt = self.get_closest_point(point)

        # create an array of boolean values for filtering points which exactly match the given coordinates
        is_x = drv_pos.X = closest_track_pnt.X
        is_y = drv_pos.Y = closest_track_pnt.Y
        is_closest_pnt = is_x and is_y

        # there may be multiple points from different laps with the given coordinates
        # therefore an estimated time range needs to be provided
        res_pnts = drv_pos[is_closest_pnt]
        for p in res_pnts:
            if time_range_start <= p.Date <= time_range_end:
                return p.Date

        return None

    def interpolate_pos_from_time(self, drv, query_date):
        """Calculate the position of a driver at any given date.

        :param drv: The number of the driver as a string
        :type drv: str
        :param query_date: The date you're interested in (pandas.Timestamp compatible)
        :return: TrackPoint
        """
        # use linear interpolation to determine position at arbitrary time
        drv_pos = self._pos_data[drv]  # get DataFrame for driver

        closest = drv_pos.iloc[(drv_pos['Date'] - query_date).abs().argsort()[:2]]

        # verify both points are valid unique track points
        if not (self.lazy_is_track_point(closest.iloc[0]['X'], closest.iloc[0]['Y']) and
                self.lazy_is_track_point(closest.iloc[1]['X'], closest.iloc[1]['Y'])):

            return None

        delta_t = closest.iloc[1]['Date'] - closest.iloc[0]['Date']
        delta_x = closest.iloc[1]['X'] - closest.iloc[0]['X']
        delta_y = closest.iloc[1]['Y'] - closest.iloc[0]['Y']
        interp_delta_t = query_date - closest.iloc[0]['Date']

        interp_x = closest.iloc[0]['X'] + delta_x * interp_delta_t / delta_t
        interp_y = closest.iloc[0]['Y'] + delta_y * interp_delta_t / delta_t

        return TrackPoint(interp_x, interp_y)


def min_index(_iterable):
    """Return the index of the minimum value in an iterable"""
    return _iterable.index(min(_iterable))


def max_index(_iterable):
    """Return the index of the minimum value in an iterable"""
    return _iterable.index(max(_iterable))


def reject_outliers(data, *secondary, m=2.):
    """Reject outliers from a numpy array.

    Calculates the deviation of each value from the median of the arrays values. Then calculates the median of
    all deviations. If a values deviation is greater than m times the median deviation, it is removed.
    An arbitrary number of additional arrays can be passed to this function. For each value that is removed
    from the reference array, the value at the corresponding index is removed from the other arryays."""
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.

    ret_secondary = list()
    for i in range(len(secondary)):
        ret_secondary.append(secondary[i][s < m])

    return data[s < m], *ret_secondary


def dump_raw_data(year, gp, event):
    session = core.get_session(year, gp, event)
    pos = api.position(session.api_path)
    tel = api.car_data(session.api_path)
    laps_data, stream_data = api.timing_data(session.api_path)

    for var, fname in zip((session, pos, tel, laps_data, stream_data), ('session', 'pos', 'tel', 'laps_data', 'stream_data')):
        with open("var_dumps/" + fname, "wb") as fout:
            pickle.dump(var, fout)


class AdvancedSyncSolver:
    """Advanced Data Synchronization and Determination of Sectors and Start/Finish Line Position
        assumptions
          - a session is always started on a full minute
              --> should be able to do without but it is easier for now

        conditions for syncing data
          - the start/finish line needs to be in a fixed place (x/y coordinates)
          - last lap start time + lap duration = current lap start time

        possible issues
          - lap and sector times are reported with what seems to be a +-0.5s accuracy
              with no further information about this process, it has to be assumed that a lap/sector time can be reported with
              an earlier or later time than its correct time (correct time = the time it was actually set at)
          - inaccuracies due to only ms precision --> max error ~50ms after the race; probably not that critical
          - laps with pit stops --> skip laps with pit in or pit out for now; only add the lap times
          - there is no fixed start time which is the sme for every driver --> maybe use race end timing?

        possible further sources of data
          - race result time between drivers for fixed values at the end too

        approach for now
          - get min/max values for start finish position from the first coarse synchronization
          - iterate over this range in small increments
              - always skip first lap
              - from selected position, interpolate a lap start time
              - add all lap times up to get a lap start time for each
              - interpolate start/finish x/y for each lap which does not have pit in or pit out
          - calculate metrics after each pass
              - arithmetic mean of x and y
              - standard deviation of x and y
              --> plot metrics
        """
    def __init__(self, track, telemetry_data, position_data, laps_data, processes=1):
        """Initiate the solver.

        :param track: Track class for this session. The track map needs to be generated beforehand!
        :type track: Track
        :param telemetry_data: Car telemetry data from the fastf1 api as returned by api.car_data
        :type telemetry_data: dict
        :param position_data: Car position data from the fastf1 api as returned by api.position
        :type position_data: dict
        :param laps_data: Lap data from the fastf1 api as returned by api.timing_data
        :type laps_data: pandas.DataFrame
        :param processes: Specifies the number of worker subprocesses where the actual data processing takes place.
            The total number of python processes will be higher but the worker processes will be the only ones which
            create significant cpu usage. One worker will approximately utilize one cpu core to 100%. Never specify
            the use of more processes than you have (virtual) cores in your system. Recommended is one or two processes
            less than the number of cores.
        :type processes: int
        """
        self.track = track
        self.tel = telemetry_data
        self.pos = position_data
        self.laps = laps_data

        self.results = dict()

        self.conditions = list()

        self.manager = None
        self.task_queue = None
        self.result_queue = None
        self.command_queue = None
        self.number_of_processes = processes
        self.subprocesses = list()

        self.drivers = None
        self.session_start_date = None
        self.point_range = list()

    def setup(self):
        """Do some one-off calculations. This needs to be called before solve() can be _run."""
        self.drivers = list(self.tel.keys())

        # calculate the start date of the session
        some_driver = self.drivers[0]  # TODO to be sure this should be done with multiple drivers
        self.session_start_date = self.pos[some_driver].head(1).Date.squeeze().round('min')

        # get all current start/finish line positions
        self.point_range = self._get_start_line_range()

    def _wait_for_results(self):
        """Wait for all processes to send their results through the result queue.
        Then results are then joined together and returned. This function blocks until all results have been received.
        This also means that all processes are guaranteed to be in an idle state when it returns."""

        idle_count = 0
        results = dict()

        while idle_count != self.number_of_processes:
            res = self.result_queue.get()

            # res is a dictionary containing {name: {key: list(), key: list(), ...}, name: ...}
            # join all lists together per key
            for name in res.keys():
                # this key exists in the joined results; extend all the existing lists with the new values
                if name in results.keys():
                    for key in res[name].keys():
                        results[name][key].extend(res[name][key])

                # this key does not yet exist; create it
                else:
                    results[name] = res[name]

            idle_count += 1

        return results

    def _queue_return_command(self):
        """Queue as many 'return' commands as there are processes. When a process receives this commands, it will return its calculation
        results and go into an idle state. During idle it will wait for further commands passed through the result_queue."""
        for _ in range(self.number_of_processes):
            self.task_queue.put('return')

    def _exit_all(self):
        """Queue as many exit commands on the command queue as there are processes."""
        for _ in range(self.number_of_processes):
            self.command_queue.put('exit')

    def _resume_all(self):
        """Queue as many resume commands on the command queue as there are processes."""
        for _ in range(self.number_of_processes):
            self.command_queue.put('resume')

    def _join_all(self):
        """Join all processes."""
        for process in self.subprocesses:
            process.join()

    def solve(self):
        """Main solver function which starts all the processing."""

        # data which the processes need
        shared_data = {'track': self.track,
                       'laps': self.laps,
                       'pos': self.pos,
                       'session_start_date': self.session_start_date}

        for cond in self.conditions:
            cond.set_data(shared_data)

        # each condition needs to be calculated for each driver
        # create a queue and populate it with (condition, driver) pairs
        self.manager = Manager()
        self.task_queue = self.manager.Queue()  # main -> subprocess: holds all tasks and the commands for returning the results
        self.result_queue = self.manager.Queue()  # subprocess -> main: return results
        self.command_queue = self.manager.Queue()  # main -> subprocess: processes block on this queue while idle waiting for a command

        self.subprocesses = list()

        print("Starting processes...")
        # create and start the processes
        for _ in range(self.number_of_processes):
            p = SolverSubprocess(self.task_queue, self.result_queue, self.command_queue, self.conditions)
            p.start()
            self.subprocesses.append(p)

        for condition in self.conditions:
            self.results[condition.name] = dict()

        print("Starting calculations...")
        start_time = time.time()  # start time for measuring _run time

        cnt = 0
        print(len(self.point_range))
        for test_point in self.point_range[0::3]:
            cnt += 1
            print(cnt)  # simplified progress report

            # Create tasks: one task consists of a condition, driver and test point
            # Do one calculation _run per test point. The results for this point are then collected and the next _run for teh next point is done.
            # Per calculation _run 'number of conditions' * 'number of drivers' = 'number of tasks'

            for condition in self.conditions:
                for driver in self.drivers:
                    # the list of conditions is passed to the process when it is created; only pass the index for a condition because sending whole
                    # classes through the queue is inefficient
                    c_index = self.conditions.index(condition)
                    self.task_queue.put((c_index, driver, test_point))
                    # each process can now fetch an item from the queue and calculate the condition for the specified driver

            # add return commands to task queue so that all processes will return their results and go to idle when the end of the queue is reached
            self._queue_return_command()
            # wait until all processes have finished processing the tasks and have returned their results
            res = self._wait_for_results()

            for condition in self.conditions:
                c_index = self.conditions.index(condition)

                proc_res = condition.generate_results(res[c_index], test_point)

                for key in proc_res.keys():
                    if key in self.results[condition.name].keys():
                        self.results[condition.name][key].append(proc_res[key])
                    else:
                        self.results[condition.name][key] = [proc_res[key], ]

            self._resume_all()  # send a resume command to all processes; they will block on the empty task queue until task are added

        # all tasks have been calculated
        print('Finished')
        print('Took:', time.time() - start_time)

        self._queue_return_command()  # queue a return command; processes currently only take exit commands while in idle state
        self._wait_for_results()  # wait for the processes to go to idle state
        self._exit_all()  # send exit command to all processes
        self._join_all()  # wait for all processes to actually exit

    def add_condition(self, condition, *args, **kwargs):
        """Add a condition class to the solver. Currently there is no check against adding duplicate conditions. Conditions can also not
        be removed again."""
        cond_inst = condition(*args, **kwargs)  # create an instance of the condition and add it to the list of solver conditions
        self.conditions.append(cond_inst)

    def _get_start_line_range(self):
        """Calculate a range of coordinates for a possible position of the start/finish line. This is done based
        on the existing lap data from the api. Extreme outliers are removed from the range of coordinates.

        :return: Two numpy arrays of x and y coordinates respectively
        """
        # find the highest and lowest x/y coordinates for the current start/finish line positions
        # positions in plural; the preliminary synchronization is not perfect
        x_coords = list()
        y_coords = list()
        usable_laps = 0  # for logging purpose

        for drv in self.drivers:
            is_drv = (self.laps.Driver == drv)  # create a list of booleans for filtering laps_data by current driver
            drv_total_laps = self.laps[is_drv].NumberOfLaps.max()  # get the current drivers total number of laps in this session

            for _, lap in self.laps[is_drv].iterrows():
                # first lap, last lap, in-lap, out-lap and laps with no lap number are skipped
                # data of these might be unreliable or imprecise
                if (pd.isnull(lap.NumberOfLaps) or
                        lap.NumberOfLaps in (1, drv_total_laps) or
                        not pd.isnull(lap.PitInTime) or
                        not pd.isnull(lap.PitOutTime)):

                    continue

                else:
                    approx_lap_end_date = self.session_start_date + lap.Time  # start of the session plus time at which the lap was registered (approximately end of lap)
                    end_pnt = self.track.interpolate_pos_from_time(drv, approx_lap_end_date)
                    x_coords.append(end_pnt.x)
                    y_coords.append(end_pnt.y)

                    usable_laps += 1

        print("{} usable laps".format(usable_laps))

        # there will still be some outliers; it's only very few though
        # so... statistics to the rescue then but allow for very high deviation as we want a long range of possible points for now
        # we only want to sort out the really far away stuff
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        x_coords, y_coords = reject_outliers(x_coords, y_coords, m=100.0)  # m defines the threshold for outliers; very high here
        print("Rejected {} outliers".format(usable_laps - len(x_coords)))

        points = list()
        index_on_track = list()
        for x, y in zip(x_coords, y_coords):
            point = track.get_closest_point(TrackPoint(x, y))
            points.append(point)
            index_on_track.append(track.sorted_points.index(point))

        point_a = points[min_index(index_on_track)]
        point_b = points[max_index(index_on_track)]

        point_range = track.get_points_between(point_a, point_b, short=True, include_ref=True)

        print("Searching for start/finish line in range x={},y={} | x={}, y={}".format(point_a.x, point_a.y, point_b.x, point_b.y))

        return point_range


class SolverSubprocess:
    """This class represents a single subprocess/worker.
    It will store all calculation results until the main process request them. The results will then be send
    trough a queue to the main process. The main process will join all results from all subprocesses."""
    def __init__(self, task_queue, result_queue, command_queue, conditions):
        """Initiate the subprocess.

        :param task_queue: main -> subprocess: tasks and command for returning data
        :type task_queue: multiprocessing.Queue
        :param result_queue: subprocess -> main: return calculation results
        :type result_queue: multiprocessing.Queue
        :param command_queue: main -> subprocess: commands for resume or exit; process will block on this queue while idling
        :type command_queue: multiprocessing.Queue
        :param conditions: List of all conditions added to the solver. Each task will contain an index which references a condition from this list.
        :type conditions: list
        """
        # use a dictionary to hold all results from processed conditions
        # multiple processes can process the same condition for different drivers simultaneously
        # therefore it is not safe to immediately save the results in the condition class
        # instead, the subprocess collects all results from the calculations from one _run (one iteration)
        # they are stored in the dictionary and teh condition's index is used as a key
        # when all subprocesses are finished, the results are collected and the conditions are updated
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._command_queue = command_queue
        self._conditions = conditions
        self._results = dict()

        self._process = Process(target=self._run)

    def start(self):
        """Start the process. Wraps multiprocessing.Process.start"""
        self._process.start()

    def join(self):
        """Wait for this process to join. Wraps multiprocessing.Process.join"""
        self._process.join()

    def _add_result(self, name, data):
        """Add a single calculation result."""
        if name not in self._results.keys():
            self._results[name] = data
        else:
            for key in data.keys():
                self._results[name][key].extend(data[key])

    def _run(self):
        """This is the main function which will loop for the lifetime of the process.
        It receives tasks and commands from the main process and calculates the results."""

        while True:
            task = self._task_queue.get()

            if task == 'return':
                # print('Going into idle', self)
                # return the calculation results and wait for commands ("idle")
                self._result_queue.put(self._results)
                self._results = dict()  # delete all results after they have been returned

                cmd = self._command_queue.get()
                if cmd == 'exit':
                    print("Exiting", self)
                    return

                elif cmd == 'resume':
                    # print("Resuming", self)
                    continue

            # print('New task', self)

            # process the received task
            # a task consists of a condition which is to be calculated, a driver to calculate if for and a test point for
            # a probable start/finish line position
            c_index, drv, point = task
            condition = self._conditions[c_index]  # get the condition from its index
            res = condition.for_driver(drv, point)  # calculate the condition and store the results
            self._add_result(c_index, res)


class BaseCondition:
    """A base class for all solver conditions.

    This class cannot be used directly but needs to be subclassed by and actual condition class."""
    def __init__(self):
        self.data = None

    def set_data(self, data):
        """
        :param data: Dictionary containing data which needs to be accessible when a subprocess calculates the condition.
        :type data: dict
        """
        self.data = data

    def for_driver(self, drv, test_point):
        """This function needs to be reimplemented by the subclass.

        :param drv: The number of the driver (as a string) for which the condition is to be calculated.
        :type drv: string
        :param test_point: A point for a possible start/finish line position
        :type test_point: TrackPoint
        """
        pass

    def generate_results(self, results, test_point):
        pass


class SectorBorderCondition(BaseCondition):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def _get_test_date(self, lap, drv, test_point):
        approx_time = self.data['session_start_date'] + lap.Time
        # now we have an approximate time for the end of the lap and we have test_x/test_y which is not unique track point
        # to get an exact time at which the car was at test_point, define a window of +-delta_t around approx_time
        delta_t = pd.to_timedelta(10, "s")
        t_start = approx_time - delta_t
        t_end = approx_time + delta_t
        pos_range = self.data['pos'][drv].query("@t_start < Date < @t_end")
        # search the two points in this range which are closest to test_point
        pos_distances = list()
        neg_distances = list()
        pos_points = list()
        neg_points = list()
        for _, row in pos_range.iterrows():
            pnt = TrackPoint(row.X, row.Y, row.Date)
            dist = test_point.get_sqr_dist(pnt)
            if pnt.x < test_point.x:
                pos_distances.append(dist)
                pos_points.append(pnt)
            else:
                neg_distances.append(dist)
                neg_points.append(pnt)

        # make sure that there are points before and after this one
        if (not neg_distances) or (not pos_distances):
            return None

        # distances, points = zip(*sorted(zip(distances, points)))  # sort distances and sort point_range exactly the same way
        p_a = pos_points[min_index(pos_distances)]
        p_b = neg_points[min_index(neg_distances)]

        dist_a_b = sqrt(p_a.get_sqr_dist(p_b))
        dist_test_a = sqrt(p_a.get_sqr_dist(test_point))

        # interpolate the time for test_point from those two points
        test_date = p_a.date + (p_b.date - p_a.date) * dist_test_a / dist_a_b
        return test_date

    def for_driver(self, drv, test_point):
        pass

    def generate_results(self, data, test_point):
        # process results
        x_series = pd.Series(data['x'])
        y_series = pd.Series(data['y'])

        result = {
            'mean_x': x_series.mean(),
            'mean_y': y_series.mean(),
            'mad_x': x_series.mad(),
            'mad_y': y_series.mad(),
            'tx': test_point.x,
            'ty': test_point.y
        }

        return result


class StartFinishCondition(SectorBorderCondition):
    """Solver condition for constant start/finish line position.

    How this condition works:
    Subtract the last lap time from the test point. (Yes subtract time from position... fancy shit going on here)
    If the test point is the actual start finish line position, this should result in the same point again. And this should
    be the case for each lap and driver.
    If the test point is not the actual start/finish line position the variation in driving between laps should cause
    a variation of the result.
    The test point at which there is the least variance in the result is deemed the correct position (simplified).
    """
    name = "StartFinish"

    def __init__(self, *args, **kwargs):
        super().__init__()

    def for_driver(self, drv, test_point):
        """ Calculate the condition for a driver and test point.

        :param drv: The driver for which to calculate the condition (driver number as a string)
        :type drv: string
        :param test_point: Start/finsih line position (test) for which to calculate the condition
        :type test_point: TrackPoint
        :return: [results x, results y] where results_* is a list of values containing the results for each lap
        """
        is_drv = (self.data['laps'].Driver == drv)
        drv_last_lap = self.data['laps'][is_drv].NumberOfLaps.max()  # get the last lap of this driver

        res_x = list()
        res_y = list()

        for _, lap in self.data['laps'][is_drv].iterrows():
            # first lap, last lap, in-lap, out-lap and laps with no lap number are skipped
            if (pd.isnull(lap.NumberOfLaps) or
                    lap.NumberOfLaps in (1, drv_last_lap) or
                    not pd.isnull(lap.PitInTime) or
                    not pd.isnull(lap.PitOutTime)):

                continue

            else:
                test_date = self._get_test_date(lap, drv, test_point)
                if not test_date:
                    continue

                # calculate start date for last lap and get position for that date
                last_lap_start = test_date - lap.LastLapTime
                lap_start_point = self.data['track'].interpolate_pos_from_time(drv, last_lap_start)
                # add point coordinates to list of results for this pass
                res_x.append(lap_start_point.x)
                res_y.append(lap_start_point.y)

        return {'x': res_x, 'y': res_y}

    def generate_results(self, data, test_point):
        # process results
        x_series = pd.Series(data['x'])
        y_series = pd.Series(data['y'])

        result = {
            'mean_x': x_series.mean(),
            'mean_y': y_series.mean(),
            'mad_x': x_series.mad(),
            'mad_y': y_series.mad(),
            'tx': test_point.x,
            'ty': test_point.y
        }

        return result


class Sector23Condition(SectorBorderCondition):
    """Solver condition for constant sector2/sector3 border position.

    How this condition works:
    Subtract the last 3rd sector time from the test point.
    If the test point is the actual start finish line position, the sector border should be the same for each lap and driver.
    Basically the same as StartFinishCondition.
    """
    name = "Sector23"

    def __init__(self, *args, **kwargs):
        super().__init__()

    def for_driver(self, drv, test_point):
        """ Calculate the condition for a driver and test point.

        :param drv: The driver for which to calculate the condition (driver number as a string)
        :type drv: string
        :param test_point: Start/finish line position (test) for which to calculate the condition
        :type test_point: TrackPoint
        :return: [results x, results y] where results_* is a list of values containing the results for each lap
        """
        is_drv = (self.data['laps'].Driver == drv)
        drv_last_lap = self.data['laps'][is_drv].NumberOfLaps.max()  # get the last lap of this driver

        res_x = list()
        res_y = list()

        for _, lap in self.data['laps'][is_drv].iterrows():
            # first lap, last lap, in-lap, out-lap and laps with no lap number are skipped
            if (pd.isnull(lap.NumberOfLaps) or
                    lap.NumberOfLaps in (1, drv_last_lap) or
                    not pd.isnull(lap.PitInTime) or
                    not pd.isnull(lap.PitOutTime)):

                continue

            else:
                test_date = self._get_test_date(lap, drv, test_point)
                if not test_date:
                    continue

                # calculate start date for last sector 3 and get position for that date
                last_sector3_start = test_date - lap.Sector3Time
                lap_sector3_point = self.data['track'].interpolate_pos_from_time(drv, last_sector3_start)
                # add point coordinates to list of results for this pass
                res_x.append(lap_sector3_point.x)
                res_y.append(lap_sector3_point.y)

        return {'x': res_x, 'y': res_y}

    def generate_results(self, data, test_point):
        # process results
        x_series = pd.Series(data['x'])
        y_series = pd.Series(data['y'])

        result = {
            'mean_x': x_series.mean(),
            'mean_y': y_series.mean(),
            'mad_x': x_series.mad(),
            'mad_y': y_series.mad(),
            'tx': test_point.x,
            'ty': test_point.y
        }

        return result


class Sector12Condition(SectorBorderCondition):
    """Solver condition for constant sector1/sector2 border position.

    How this condition works:
    Subtract the last 3rd sector time and 2nd sector time from the test point.
    If the test point is the actual start finish line position, the sector border should be the same for each lap and driver.
    Basically the same as StartFinishCondition.
    """
    name = "Sector12"

    def __init__(self, *args, **kwargs):
        super().__init__()

    def for_driver(self, drv, test_point):
        """ Calculate the condition for a driver and test point.

        :param drv: The driver for which to calculate the condition (driver number as a string)
        :type drv: string
        :param test_point: Start/finish line position (test) for which to calculate the condition
        :type test_point: TrackPoint
        :return: [results x, results y] where results_* is a list of values containing the results for each lap
        """
        is_drv = (self.data['laps'].Driver == drv)
        drv_last_lap = self.data['laps'][is_drv].NumberOfLaps.max()  # get the last lap of this driver

        res_x = list()
        res_y = list()

        for _, lap in self.data['laps'][is_drv].iterrows():
            # first lap, last lap, in-lap, out-lap and laps with no lap number are skipped
            if (pd.isnull(lap.NumberOfLaps) or
                    lap.NumberOfLaps in (1, drv_last_lap) or
                    not pd.isnull(lap.PitInTime) or
                    not pd.isnull(lap.PitOutTime)):

                continue

            else:
                test_date = self._get_test_date(lap, drv, test_point)
                if not test_date:
                    continue

                # calculate start date for last sector 2 and get position for that date
                last_sector2_start = test_date - lap.Sector3Time - lap.Sector2Time
                lap_sector2_point = self.data['track'].interpolate_pos_from_time(drv, last_sector2_start)
                # add point coordinates to list of results for this pass
                res_x.append(lap_sector2_point.x)
                res_y.append(lap_sector2_point.y)

        return {'x': res_x, 'y': res_y}

    def generate_results(self, data, test_point):
        # process results
        x_series = pd.Series(data['x'])
        y_series = pd.Series(data['y'])

        result = {
            'mean_x': x_series.mean(),
            'mean_y': y_series.mean(),
            'mad_x': x_series.mad(),
            'mad_y': y_series.mad(),
            'tx': test_point.x,
            'ty': test_point.y
        }

        return result


class AllSectorBordersCondition(SectorBorderCondition):
    """Solver condition for constant sector1/sector2 border position.

    How this condition works:
    Subtract the last 3rd sector time and 2nd sector time from the test point.
    If the test point is the actual start finish line position, the sector border should be the same for each lap and driver.
    Basically the same as StartFinishCondition.
    """
    name = "AllSectors"

    def __init__(self, *args, **kwargs):
        super().__init__()

    def for_driver(self, drv, test_point):
        """ Calculate the condition for a driver and test point.

        :param drv: The driver for which to calculate the condition (driver number as a string)
        :type drv: string
        :param test_point: Start/finish line position (test) for which to calculate the condition
        :type test_point: TrackPoint
        :return: [results x, results y] where results_* is a list of values containing the results for each lap
        """
        is_drv = (self.data['laps'].Driver == drv)
        drv_last_lap = self.data['laps'][is_drv].NumberOfLaps.max()  # get the last lap of this driver

        res = {'x1': list(), 'y1': list(), 'x2': list(), 'y2': list(), 'x3': list(), 'y3': list()}

        for _, lap in self.data['laps'][is_drv].iterrows():
            # first lap, last lap, in-lap, out-lap and laps with no lap number are skipped
            if (pd.isnull(lap.NumberOfLaps) or
                    lap.NumberOfLaps in (1, drv_last_lap) or
                    not pd.isnull(lap.PitInTime) or
                    not pd.isnull(lap.PitOutTime)):

                continue

            else:
                test_date = self._get_test_date(lap, drv, test_point)
                if not test_date:
                    continue

                # sector 1/2
                last_sector2_start = test_date - lap.Sector3Time - lap.Sector2Time
                lap_sector2_point = self.data['track'].interpolate_pos_from_time(drv, last_sector2_start)
                # add point coordinates to list of results for this pass
                res['x2'].append(lap_sector2_point.x)
                res['y2'].append(lap_sector2_point.y)

                # sector 2/3
                last_sector3_start = test_date - lap.Sector3Time
                lap_sector3_point = self.data['track'].interpolate_pos_from_time(drv, last_sector3_start)
                res['x3'].append(lap_sector3_point.x)
                res['y3'].append(lap_sector3_point.y)

                # start/finish
                last_lap_start = test_date - lap.LastLapTime
                lap_start_point = self.data['track'].interpolate_pos_from_time(drv, last_lap_start)
                res['x1'].append(lap_start_point.x)
                res['y1'].append(lap_start_point.y)

        return res

    def generate_results(self, data, test_point):
        # process results
        x1_series = pd.Series(data['x1'])
        y1_series = pd.Series(data['y1'])
        x2_series = pd.Series(data['x2'])
        y2_series = pd.Series(data['y2'])
        x3_series = pd.Series(data['x3'])
        y3_series = pd.Series(data['y3'])

        result = {
            'mean_x1': x1_series.mean(),
            'mean_y1': y1_series.mean(),
            'mad_x1': x1_series.mad(),
            'mad_y1': y1_series.mad(),
            'mean_x2': x2_series.mean(),
            'mean_y2': y2_series.mean(),
            'mad_x2': x2_series.mad(),
            'mad_y2': y2_series.mad(),
            'mean_x3': x3_series.mean(),
            'mean_y3': y3_series.mean(),
            'mad_x3': x3_series.mad(),
            'mad_y3': y3_series.mad(),
            'tx': test_point.x,
            'ty': test_point.y
        }

        return result


if __name__ == '__main__':
    session = pickle.load(open("var_dumps/session", "rb"))
    pos = pickle.load(open("var_dumps/pos", "rb"))
    tel = pickle.load(open("var_dumps/tel", "rb"))
    laps_data = pickle.load(open("var_dumps/laps_data", "rb"))
    # track = pickle.load(open("var_dumps/track_map", "rb"))

    track = Track(pos)
    track.generate_track(visualization_frequency=250)

    solver = AdvancedSyncSolver(track, tel, pos, laps_data, processes=6)
    solver.setup()
    solver.log_setup_stats()
    solver.add_condition(StartFinishCondition)
    solver.solve()

    IPython.embed()
