# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KPI visualizer that bucketizes users in kpi sets."""

import json
import os
import pandas as pd

varnan = float('nan')


class KpiVisualizer(object):
    """
    KPI visualizer collects information of each user through parsing json_datafactory.

    The following information will be collected:
    set id, user id, image name, if all information of frame are present, left and right
    eyes status, left and right eyes occluded, left and right pupils occluded, left and right
    iris occluded
    """

    def __init__(self, kpi_sets, kpi_bucket_file, path_info):
        """
        Initializes kpi visualizer.

        Args:
        kpi_sets (list of set ids(str)): kpi sets to visualize
        kpi_bucket_file (str): a csv file containing users information of each kpi set
        path_info (dict of lists(str)): path info for searching sets across multiple cosmos
        """

        if kpi_bucket_file is None or kpi_bucket_file == '' or \
                '.csv' not in kpi_bucket_file or not os.path.isfile(kpi_bucket_file):
            self._kpi_bucket_file = None
        else:
            self._kpi_bucket_file = kpi_bucket_file

        self.csvAttrColumns = ['set_id', 'user_id', 'region_name', 'image_name', 'valid',
                               'leye_status', 'reye_status', 'leyelid_occl', 'reyelid_occl',
                               'lpupil_occl', 'rpupil_occl', 'liris_occl', 'riris_occl', 'glasses',
                               'race', 'gender', 'tv_size', 'cam_position',
                               'left_eye_glare_status', 'left_pupil_glare_status',
                               'right_eye_glare_status', 'right_pupil_glare_status']

        self.csvTable = pd.DataFrame(columns=self.csvAttrColumns)

        self._root = path_info['root_path']
        self.gcosmos = path_info['set_directory_path']
        self.kpisets = []
        for sets_list in kpi_sets:
            self.kpisets.append(sets_list.split(' '))

        self.csvFile = 'allkpi_dfstates_userattr.csv'

    def __call__(self, output_path, write_csv=False):
        """
        Parses csv file to collect information, if csv is not present or invalid.

        parse data sets to generate new table
        """

        if self._kpi_bucket_file is None:
            print('Invalid csv file entered, regenerating new csv now...')

            if len(self.gcosmos) != len(self.kpisets):
                raise ValueError('Expected length of set_directory_path and'
                                 'visualize_set_id to be same, received: {}, {}'.
                                 format(len(self.gcosmos),
                                        len(self.kpisets)))

            self.generate_attrDF(self.gcosmos, self.kpisets)
            self.csvTable = self.addAttributes(self.csvTable)

        else:
            self.csvTable = pd.read_csv(self._kpi_bucket_file)
            print('KPI Sanity: {}'.format(len(self.csvTable.index)))
            for gcosmos, sets_list in zip(self.gcosmos, self.kpisets):
                self.addAdditionalSets(self.csvTable, gcosmos, sets_list)

            print('KPI Sanity: {}'.format(len(self.csvTable.index)))
            if len(self.csvTable.columns) < len(self.csvAttrColumns):
                self.csvTable = self.addAttributes(self.csvTable)

        if write_csv:
            file_path = os.path.join(output_path, self.csvFile)
            self.csvTable.to_csv(path_or_buf=file_path, mode='w+')
            print('Attributes csv is generated: ', file_path)

        return self.csvTable

    def addAdditionalSets(self, df, cosmos_path, kpi_sets):
        """Finds missing data set from attribute file and append to it."""

        additionalSets = []
        for set_id in kpi_sets:
            df_set = df[df['set_id'] == set_id]
            if len(df_set.index) == 0:
                additionalSets.append(set_id)
                print('Appending additional set {} to attributes csv'.format(set_id))

        self.generate_attrDF([cosmos_path], [additionalSets])

    def addAttributes(self, df):
        """Add attributes to data frame."""
        df = df.astype(str)
        for index, row in df.iterrows():
            for root_dir, kpi_sets in zip(self.gcosmos, self.kpisets):
                if row['set_id'] in kpi_sets:
                    _root_directory = root_dir

            attr = self.__getAttributes(_root_directory, row['set_id'], row['user_id'])
            df.at[index, 'glasses'] = str(attr['glasses']) if 'glasses' in attr else str('NA')
            df.at[index, 'race'] = str(attr['race']) if 'race' in attr else str('NA')
            df.at[index, 'gender'] = str(attr['gender']) if 'gender' in attr else str('NA')
            df.at[index, 'tv_size'] = str(attr['tv_size']) if 'tv_size' in attr else str('NA')
            df.at[index, 'cam_position'] = str(attr['cam_position']) \
                if 'cam_position' in attr else str('NA')
        return df

    def __getAttributes(self, root_dir, setpath, username):
        attributes = {}
        setpath = os.path.join(root_dir, setpath)
        setpath = setpath.replace('postData', 'orgData')
        if 'copilot.cosmos10' in setpath:
            glasses_ind = 5
        elif 'driveix.cosmos639' in setpath:
            glasses_ind = 11

        for fname in os.listdir(setpath):
            if 'summary' in fname and fname.endswith('.csv'):
                with open(os.path.join(setpath, fname)) as fp:
                    for line in fp:
                        if 'user' in line.lower() and 'Badge No' not in line and username in line:
                            attributes['hash'] = username
                            parts = line.strip('\n').split(',')
                            if 'yes' in parts[glasses_ind].lower() or \
                                    'glasses' in parts[glasses_ind].lower():
                                attributes['glasses'] = str('Yes')
                            else:
                                attributes['glasses'] = str('No')
                            attributes['gender'] = str('Female') \
                                if 'female' in line.lower() else str('Male')
                            attributes['race'] = str(parts[4])
                            break

        camfile = os.path.join(setpath, 'Config', 'setup.csv')
        if not os.path.exists(camfile):
            camfile = os.path.join(setpath, 'setup.csv')

        if os.path.isfile(camfile):
            with open(camfile) as fp:
                for line in fp:
                    if "tv_size" not in line:
                        parts = line.strip('\n').split(',')
                        attributes['tv_size'] = str(parts[0])
                        attributes['cam_position'] = str(parts[-2])
        else:
            # Note that most sets do not have setup.csv to parse!
            # print("Camfile not found {}".format(camfile))
            attributes['tv_size'] = 'NA'
            attributes['cam_position'] = "NA"
        return attributes

    def generate_attrDF(self, cosmos_path_list, kpi_sets):
        """Generate attributes for data frame."""

        for root_dir, set_ids in zip(cosmos_path_list, kpi_sets):
            for set_id in set_ids:
                print('Generating data frame for {}'.format(set_id))
                self.parse_set_jsons(self._root, root_dir, set_id)

    def parse_set_jsons(self, root_path, root_dir, set_id):
        """"Parse json files in each set."""

        if root_path is None:
            root_path = ''

        json_datafactory = os.path.join(root_path, root_dir, set_id)
        json_datafactory = json_datafactory.replace('postData', 'orgData')

        if os.path.exists(os.path.join(json_datafactory, 'json_datafactory_v2')):
            json_datafactory = os.path.join(json_datafactory, 'json_datafactory_v2')
        elif os.path.exists(os.path.join(json_datafactory, 'json_datafactory')):
            json_datafactory = os.path.join(json_datafactory, 'json_datafactory')
        else:
            print('Cannot find json_datafactory_v2 or json_datafactory in {}'
                  .format(json_datafactory))
            # The table returned might be empty now.
            # That is ok as long as the visualization script can handle it.
            return

        jsons = os.listdir(json_datafactory)
        for _json in jsons:
            if _json.startswith('.') or 'json' not in _json or _json.endswith('.swp'):
                continue

            table = self.parse_json_file(json_datafactory, _json)
            table = pd.DataFrame(table, columns=self.csvAttrColumns)
            self.csvTable = pd.concat([self.csvTable, table], ignore_index=True, sort=True)

    def parse_json_file(self, json_datafactory, _json):
        """Parse the given json file."""

        table = []

        parts = _json.split('/')[-1].split('.json')[0].split("_")
        if "s400_KPI" not in _json:
            setid = parts[0]
            userid = parts[1]
        else:
            setid = "s400_KPI"
            userid = _json.split('/')[-1].split("_", 2)[-1].split('.json')[0]

        _json_file = os.path.join(json_datafactory, _json)
        text = open(_json_file, "r").read()
        try:
            _json_contents = json.loads(text)
        except Exception:
            print("Failed to load {}".format(str(_json_file)))
            return None

        for section in _json_contents:
            valid = True
            try:
                filename_split = section['filename'].split("/")
                frame_name = filename_split[-1]
                region_name = filename_split[-2]
                if not region_name.startswith('region_'):
                    # Legacy flat structure, no regions.
                    region_name = ''
            except Exception:
                print('Failed to read filename, skipping...')
                continue

            if 'png' not in frame_name and 'jpg' not in frame_name:
                print("filename doesn't appear to be an image {}".format(frame_name))
                continue
            if len(section['annotations']) == 0:
                print("Image contains zero annotations {} {}".format(frame_name, _json_file))
                continue
            if len(section['annotations']) == 1:
                print("skipping invalid image {}".format(section['filename']))
                valid = False
                row = [setid, userid, region_name, frame_name, valid, "invalid", "invalid",
                       varnan, varnan, varnan, varnan, varnan, varnan]
                table.append(row)
                continue

            for chunk in section['annotations']:
                numLandmarks = 104
                x = ['0'] * numLandmarks
                y = ['0'] * numLandmarks
                tags = [0] * numLandmarks
                lastidx = 0

                if chunk.get('class') is None:
                    continue
                if 'eyes' in chunk['class'].lower():
                    # Check for eye status. Switch left-right convention
                    # Data factory labels left from labellers perspective (not users)
                    reye_status = chunk['l_status']
                    leye_status = chunk['r_status']

                # To support Old labelling. sigh
                elif 'eye' in chunk['class'].lower():
                    leye_status = reye_status = 'open' if 'open' \
                                                          in chunk['class'].lower() else 'closed'

                # Obtain fiducial information
                elif 'fiducialpoints' in chunk['class'].lower():
                    for point in chunk:
                        if 'version' in point or 'class' in point or 'Poccluded' in point:
                            continue
                        landmark_pt = int(''.join(c for c in str(point) if c.isdigit()))
                        lastidx = max(landmark_pt, lastidx)

                        if 'x' in point and landmark_pt < numLandmarks:
                            x[landmark_pt - 1] = str(int(float(chunk[point])))
                        if 'y' in point and landmark_pt < numLandmarks:
                            y[landmark_pt - 1] = str(int(float(chunk[point])))
                        if 'occ' in str(point).lower() and landmark_pt <= numLandmarks:
                            tags[landmark_pt - 1] = 1

                    # Calculate occlusions
                    del (x[lastidx:])
                    del (y[lastidx:])
                    del (tags[lastidx:])
                    reyelid_occl = tags[36:41].count(1)
                    leyelid_occl = tags[42:47].count(1)
                    rpupil_occl = tags[68:72].count(1)
                    lpupil_occl = tags[72:76].count(1)
                    if lastidx > 100:
                        rIris_occlusions = tags[82:91].count(1)
                        lIris_occlusions = tags[93:102].count(1)
                    else:
                        rIris_occlusions = lIris_occlusions = varnan

            try:
                row = [setid, userid, region_name, frame_name, valid, leye_status, reye_status,
                       leyelid_occl, reyelid_occl, lpupil_occl, rpupil_occl,
                       lIris_occlusions, rIris_occlusions, varnan, varnan, varnan,
                       varnan, varnan, varnan, varnan, varnan, varnan]
                table.append(row)
            except Exception as e:
                print('Error processing: {}'.format([setid, userid, region_name, frame_name,
                                                     valid, leye_status, reye_status,
                      leyelid_occl, reyelid_occl, lpupil_occl, rpupil_occl,
                                                     lIris_occlusions, rIris_occlusions]))
                print('{}'.format(e))

        return table
