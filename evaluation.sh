

INPUT_MESH=output/scannet/scene0106_00/DSGD_D50_1/agent_0/mesh_track773.ply
python cull_mesh.py --config configs/scannet/scene0106.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose


REC_MESH=output/scannet/scene0106_00/DSGD_D50_1/agent_0/mesh_track773_cull_occlusion.ply
GT_MESH=output/scannet/scene0106_00/Centralized/agent_0/mesh_track2323_cull_occlusion.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH -3d




INPUT_MESH=output/scannet/scene0106_00/DSGD_D50_2/agent_0/mesh_track773.ply
python cull_mesh.py --config configs/scannet/scene0106.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose


REC_MESH=output/scannet/scene0106_00/DSGD_D50_2/agent_0/mesh_track773_cull_occlusion.ply
GT_MESH=output/scannet/scene0106_00/Centralized/agent_0/mesh_track2323_cull_occlusion.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH -3d




INPUT_MESH=output/scannet/scene0106_00/DSGD_D50_3/agent_0/mesh_track773.ply
python cull_mesh.py --config configs/scannet/scene0106.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose


REC_MESH=output/scannet/scene0106_00/DSGD_D50_3/agent_0/mesh_track773_cull_occlusion.ply
GT_MESH=output/scannet/scene0106_00/Centralized/agent_0/mesh_track2323_cull_occlusion.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH -3d




INPUT_MESH=output/scannet/scene0106_00/DSGD_D50_1/agent_1/mesh_track773.ply
python cull_mesh.py --config configs/scannet/scene0106.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose


REC_MESH=output/scannet/scene0106_00/DSGD_D50_1/agent_1/mesh_track773_cull_occlusion.ply
GT_MESH=output/scannet/scene0106_00/Centralized/agent_0/mesh_track2323_cull_occlusion.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH -3d




INPUT_MESH=output/scannet/scene0106_00/DSGD_D50_2/agent_1/mesh_track773.ply
python cull_mesh.py --config configs/scannet/scene0106.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose


REC_MESH=output/scannet/scene0106_00/DSGD_D50_2/agent_1/mesh_track773_cull_occlusion.ply
GT_MESH=output/scannet/scene0106_00/Centralized/agent_0/mesh_track2323_cull_occlusion.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH -3d




INPUT_MESH=output/scannet/scene0106_00/DSGD_D50_3/agent_1/mesh_track773.ply
python cull_mesh.py --config configs/scannet/scene0106.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose


REC_MESH=output/scannet/scene0106_00/DSGD_D50_3/agent_1/mesh_track773_cull_occlusion.ply
GT_MESH=output/scannet/scene0106_00/Centralized/agent_0/mesh_track2323_cull_occlusion.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH -3d




INPUT_MESH=output/scannet/scene0106_00/DSGD_D50_1/agent_2/mesh_track773.ply
python cull_mesh.py --config configs/scannet/scene0106.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose


REC_MESH=output/scannet/scene0106_00/DSGD_D50_1/agent_2/mesh_track773_cull_occlusion.ply
GT_MESH=output/scannet/scene0106_00/Centralized/agent_0/mesh_track2323_cull_occlusion.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH -3d




INPUT_MESH=output/scannet/scene0106_00/DSGD_D50_2/agent_2/mesh_track773.ply
python cull_mesh.py --config configs/scannet/scene0106.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose


REC_MESH=output/scannet/scene0106_00/DSGD_D50_2/agent_2/mesh_track773_cull_occlusion.ply
GT_MESH=output/scannet/scene0106_00/Centralized/agent_0/mesh_track2323_cull_occlusion.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH -3d




INPUT_MESH=output/scannet/scene0106_00/DSGD_D50_3/agent_2/mesh_track773.ply
python cull_mesh.py --config configs/scannet/scene0106.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose


REC_MESH=output/scannet/scene0106_00/DSGD_D50_3/agent_2/mesh_track773_cull_occlusion.ply
GT_MESH=output/scannet/scene0106_00/Centralized/agent_0/mesh_track2323_cull_occlusion.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH -3d


