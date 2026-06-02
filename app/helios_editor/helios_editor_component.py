import os
import streamlit.components.v1 as components

_RELEASE = True # Switch to True before deployment


if not _RELEASE:
    _COMPONENT = components.declare_component(
        "helios_editor",
        url="http://localhost:5173",
    )
else:
    build_dir = os.path.join(
        os.path.dirname(__file__),
        "dist",
    )

    _COMPONENT = components.declare_component(
        "helios_editor",
        path=build_dir,
    )


def helios_editor(
        image_src=None,
        focus_objects=None,
        ai_suggestions=None,
        initial_state=None,
        pending_inset_asset=None,
        export_request_id=None,
        project_key=None,
        debug=False,
        key=None,
):
    component_value = _COMPONENT(
        imageSrc=image_src,
        focusObjectsFromStreamlit=focus_objects,
        aiSuggestions=ai_suggestions,
        initialState=initial_state,
        debug=debug,
        default={},
        key=key,
        height=1200,
        pendingInsetAsset=pending_inset_asset,
        exportRequestId=export_request_id,
        projectKey=project_key,
    )

    return component_value