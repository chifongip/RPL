# X2 Ultra runtime assets

Generated with `scripts/build_x2_assets.py` from X2_URDF-v1.3.0 simplified collisions. Head joints are fixed at zero with mass/inertia preserved. MuJoCo is converted from the same training URDF with symmetric waist-yaw stops at +/-2.2078 rad; the source vendor files are unchanged. Camera meshes bake fixed-link transforms. Vendor assets are under the accompanying Mulan PSL v2 LICENSE. Gains and HOME are selected separately in parameters.json.
