"""Push data to desq database."""
# As ai-infra does not allow git submodules or wheel installs, get desq from cosmos distribution.
import sys
sys.path.append('/home/driveix.cosmos639/desq/dist/v1.0.0')
import desq  # noqa: E402

from desq.schema import EyecropComputation, EyecropCreation, EyecropLabelling, EyestateCreation, \
                        EyestateLabelling, EyestateModel, FacecropComputation, FacecropCreation, \
                        FacecropLabelling, GazeComputation, GazeCreation, GazeoriginComputation, \
                        GazeoriginCreation, GazeoriginCreator, HeadposeComputation, \
                        HeadposeCreation, LandmarksCreation, LandmarksLabelling, LandmarksModel \
                        # noqa: E402


def get_creator(session, creator_class, creator_name, constructor_kwargs=None):
    """Return a creator instance from the database (adding to database if needed)."""
    creator = session.query(creator_class).filter_by(name=creator_name).one_or_none()
    if creator is None:
        if constructor_kwargs is None:  # So that we don't get a global dict as default param.
            constructor_kwargs = {}
        if issubclass(creator_class, desq.schema.ModelMixin):  # Model class (driveIX SDK).
            creator = creator_class(name=creator_name,
                                    uri_path="{}/{}".format(creator_name, creator_class.__name__),
                                    storage_id=3,
                                    **constructor_kwargs)
        else:  # Computation or Labelling class.
            creator = creator_class(name=creator_name, **constructor_kwargs)
        session.add(creator)
    return creator


def get_creators(session, strategy_type, landmarks_folder_name, sdklabels_folder_name):
    """Return a list of tuple(creation_class, creator, kwargs) to create prediction for.

    Each of the creators returned is already added to the session (thus has an id).
    """
    if landmarks_folder_name is not None:
        raise NotImplementedError("Writing to desq not yet supported with -landmarks-folder")
    if sdklabels_folder_name is not None:
        raise NotImplementedError("Writing to desq not yet supported with -sdklabels-folder")

    if strategy_type == 'sdk':
        eyestate_creator = get_creator(session, EyestateModel, u"SDK_v1")
        landmarks_creator = get_creator(session, LandmarksModel, u"SDK_v1",
                                        {"keypoint_count": 80})
        eyecrop_creator = get_creator(session, EyecropComputation, u"SDK_v1",
                                      {"landmarkscreator": landmarks_creator})
        facecrop_creator = get_creator(session, FacecropComputation, u"SDK_v1",
                                       {"landmarkscreator": landmarks_creator})
    else:
        eyestate_creator = get_creator(session, EyestateLabelling, u"DataFactory_v1")
        landmarks_creator = get_creator(session, LandmarksLabelling, u"DataFactory_v1",
                                        {"keypoint_count": 104})
        eyecrop_creator = get_creator(session, EyecropLabelling, u"DataFactory_v1")
        facecrop_creator = get_creator(session, FacecropLabelling, u"DataFactory_v1")
    gazeorigin_mid = get_creator(session, GazeoriginComputation, u"PnP_center_of_eyes_v1",
                                 {"origin_type": GazeoriginCreator.OriginType.CENTER_OF_EYES,
                                  "landmarkscreator": landmarks_creator})
    gazeorigin_lpc = get_creator(session, GazeoriginComputation, u"PnP_left_eye_v1",
                                 {"origin_type": GazeoriginCreator.OriginType.LEFT_EYE,
                                  "landmarkscreator": landmarks_creator})
    gazeorigin_rpc = get_creator(session, GazeoriginComputation, u"PnP_right_eye_v1",
                                 {"origin_type": GazeoriginCreator.OriginType.RIGHT_EYE,
                                  "landmarkscreator": landmarks_creator})
    return [
        (EyestateCreation, eyestate_creator, {}),
        (EyecropCreation, eyecrop_creator, {}),
        (FacecropCreation, facecrop_creator, {}),
        (GazeoriginCreation, gazeorigin_mid, {"origin_prefix": "mid_"}),
        (GazeoriginCreation, gazeorigin_lpc, {"origin_prefix": "lpc_"}),
        (GazeoriginCreation, gazeorigin_rpc, {"origin_prefix": "rpc_"}),
        (GazeCreation, get_creator(session, GazeComputation, u"PnP_center_of_eyes_v1",
                                   constructor_kwargs={"origincreator": gazeorigin_mid}),
            {"origin_postfix": ""}),
        (GazeCreation, get_creator(session, GazeComputation, u"PnP_left_eye_v1",
                                   constructor_kwargs={"origincreator": gazeorigin_lpc}),
            {"origin_postfix": "_le"}),
        (GazeCreation, get_creator(session, GazeComputation, u"PnP_right_eye_v1",
                                   constructor_kwargs={"origincreator": gazeorigin_rpc}),
            {"origin_postfix": "_re"}),
        (HeadposeCreation, get_creator(session, HeadposeComputation, u"PnP_v1"), {}),
        (LandmarksCreation, landmarks_creator, {}),
    ]


def build_creation(creation_class, frame_data_dict, origin_prefix="mid_", origin_postfix=""):
    """Return a new creation_class for the given frame_data_dict."""
    if creation_class == EyestateCreation:
        creation = EyestateCreation(
            left_eye=EyestateCreation.EyeState.from_str(frame_data_dict["label/left_eye_status"]),
            right_eye=EyestateCreation.EyeState.from_str(frame_data_dict["label/right_eye_status"])
        )
    elif creation_class == FacecropCreation:
        creation = FacecropCreation(
            x1=frame_data_dict["train/tight_facebbx_x1"],
            y1=frame_data_dict["train/tight_facebbx_y1"],
            width=frame_data_dict["train/tight_facebbx_x2"]
            - frame_data_dict["train/tight_facebbx_x1"],
            height=frame_data_dict["train/tight_facebbx_y2"]
            - frame_data_dict["train/tight_facebbx_y1"],
        )
    elif creation_class == EyecropCreation:
        creation = EyecropCreation(
            left_x1=frame_data_dict["train/lefteyebbx_x"],
            left_y1=frame_data_dict["train/lefteyebbx_y"],
            left_width=frame_data_dict["train/lefteyebbx_w"],
            left_height=frame_data_dict["train/lefteyebbx_h"],
            right_x1=frame_data_dict["train/righteyebbx_x"],
            right_y1=frame_data_dict["train/righteyebbx_y"],
            right_width=frame_data_dict["train/righteyebbx_w"],
            right_height=frame_data_dict["train/righteyebbx_h"],
        )
    elif creation_class == GazeoriginCreation:
        creation = GazeoriginCreation(
            origin_mm=(frame_data_dict["label/{}cam_x".format(origin_prefix)],
                       frame_data_dict["label/{}cam_y".format(origin_prefix)],
                       frame_data_dict["label/{}cam_z".format(origin_prefix)]),
        )
    elif creation_class == GazeCreation:
        creation = GazeCreation(
            gaze_cam_mm=(frame_data_dict["label/gaze_cam_x"],
                         frame_data_dict["label/gaze_cam_y"],
                         frame_data_dict["label/gaze_cam_z"]),
        )
        if frame_data_dict["train/valid_theta_phi"]:  # Otherwise theta/phi == None.
            creation.theta = frame_data_dict["label/theta{}".format(origin_postfix)]
            creation.phi = frame_data_dict["label/phi{}".format(origin_postfix)]
    elif creation_class == HeadposeCreation:
        creation = HeadposeCreation(
            headpose_degrees=(frame_data_dict["label/hp_pitch"],
                              frame_data_dict["label/hp_yaw"],
                              frame_data_dict["label/hp_roll"]),
        )
    elif creation_class == LandmarksCreation:
        creation = LandmarksCreation.from_flat_list(
            frame_data_dict["train/landmarks"],
            occluded=frame_data_dict["train/landmarks_occ"]
        )
    else:
        raise NotImplementedError("Unknown creation_class {}".format(creation_class))
    return creation


def write_desq(users_dict, strategy_type, landmarks_folder_name, sdklabels_folder_name,
               update=False):
    """Add the given users_dict to desq.

    If update is True, replaces any existing creations (of the same creators) by the new ones.
    """
    with desq.session_scope() as session:
        creators = get_creators(session, strategy_type, landmarks_folder_name,
                                sdklabels_folder_name)

        for user in users_dict.keys():
            print("Pushing images from user {}".format(user))
            for region in users_dict[user].keys():
                for frame in users_dict[user][region].keys():
                    frame_data_dict = users_dict[user][region][frame]

                    # Remove /home/copilot.cosmos10/ or /home/driveix.cosmos639/ from file path.
                    uri_path = frame_data_dict["train/image_frame_name"].split('/', 3)[3]
                    # Get the corresponding image from desq.
                    image = (
                        session.query(desq.schema.OriginalImage)
                        .filter_by(uri_path=uri_path).one()
                    )

                    # Add all predictions.
                    for creation_class, creator, kwargs in creators:
                        creation = build_creation(creation_class, frame_data_dict, **kwargs)
                        creation.creator_id = creator.id
                        creation.image_id = image.id
                        if update:
                            creation = session.merge(creation)
                        session.add(creation)
