using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;
using UnityEngine.Rendering;
using Unity.XR.Oculus;
using TMPro;
using UnityEditor;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;


public class MainDataRecorder : MonoBehaviour
{
    // Define some variables for program
    #region Private Variables
    OVRCameraRig cameraRig;
    private UdpClient client;
    private IPEndPoint remoteEndPoint;
    private Socket sender;
    private IPEndPoint targetEndPoint;
    #endregion

    #region Unity Inspector Variables
    // [SerializeField]
    // [Tooltip("The RawImage where the virtual depth map will be displayed.")]
    // private RawImage m_virtualDepthImage;
    [SerializeField]
    [Tooltip("Time text")]
    private TextMeshProUGUI m_TimeText;
    [SerializeField]
    public string local_ip;
    [SerializeField]
    public int listen_port = 65432;
    [SerializeField]
    public int sender_port = 12346;
    private Quaternion rotateZX = Quaternion.Euler(45f, 0f,90f);
    private Quaternion rotateZXinv = Quaternion.Euler(45f, 0f, -90f);
    private Quaternion rotateX = Quaternion.Euler(-10f, -25f,-25f);
    private Quaternion rotateXinv = Quaternion.Euler(-30f, 25f, 25f);
    private Vector3 right_pos_offset = new Vector3(0.08f, 0.01f, -0.05f);
    private Vector3 left_pos_offset = new Vector3(-0.1f, -0.04f, -0.07f);
    private Vector3 rleap_pos_offset = new Vector3(0.05f, 0.1f, 0.05f);
    private Quaternion rleap_rot_offset = Quaternion.Euler(0f, 0f, 180f);
    private float current_time = 0.0f;

    private string current_text;
    #endregion // Unity Inspector Variables

    /// <summary>
    /// Attempts to get any unassigned components.
    /// </summary>
    /// <returns>
    /// <c>true</c> if all components were satisfied; otherwise <c>false</c>.
    /// </returns>
    private bool TryGetComponents()
    {
        if (m_TimeText == null) { m_TimeText = GetComponent<TextMeshProUGUI>(); }
        return m_TimeText != null;
    }
    private bool MainLoop()
    {
        // Attempt to get the global depth texture
        // This should be a image, get a image and send via redis?
        // robot.transform.position = CoordinateFrame.current_pos;
        // robot.transform.rotation = CoordinateFrame.cum_rot;
        // robot_base.transform.position = CoordinateFrame.current_pos;
        // robot_base.transform.rotation = CoordinateFrame.cum_rot;
        // rleap.GetComponent<ArticulationBody>().TeleportRoot(robot_ee.transform.position + robot_ee.transform.rotation * rleap_rot_offset * rleap_pos_offset, 
        //                                                     robot_ee.transform.rotation * rleap_rot_offset);
        // rleap.transform.position = robot_ee.transform.position + robot_ee.transform.rotation * rleap_rot_offset * rleap_pos_offset;
        // rleap.transform.rotation = robot_ee.transform.rotation * rleap_rot_offset;
        // rleap.GetComponent<ArticulationBody>().TeleportRoot(robot_ee.transform.position + robot_ee.transform.rotation * rleap_rot_offset * rleap_pos_offset, robot_ee.transform.rotation * rleap_rot_offset);
        
        var headPose = cameraRig.centerEyeAnchor.position;
        var headRot = cameraRig.centerEyeAnchor.rotation; // Should store them separately. [w,x,y,z]
        m_TimeText.enabled = true;
        
        // Left controller on right hand, inversed
        // Vector3 rightWristPos = cameraRig.leftHandAnchor.position + cameraRig.leftHandAnchor.rotation * left_pos_offset;
        // Vector3 leftWristPos = cameraRig.rightHandAnchor.position + cameraRig.rightHandAnchor.rotation * right_pos_offset;
        // Quaternion rightWristRot = cameraRig.leftHandAnchor.rotation * rotateXinv * rotateZXinv;
        // Quaternion leftWristRot = cameraRig.rightHandAnchor.rotation * rotateX * rotateZX;
        Vector3 leftWristPos = cameraRig.leftHandAnchor.position;
        Quaternion leftWristRot = cameraRig.leftHandAnchor.rotation;

        Vector3 rightWristPos = cameraRig.rightHandAnchor.position;
        Quaternion rightWristRot = cameraRig.rightHandAnchor.rotation;

        m_TimeText.text = current_text + " rightWristPos: (" + rightWristPos.x + "," + rightWristPos.y + "," + rightWristPos.z + ")";
        // updateVisSpheres(hand, leftWristPos, leftWristRot, rightWristPos, rightWristRot);
        // if time gap > 0.05 send hand pose
        if (Time.time - current_time > 0.02)
        {
            SendHeadBimanualWristPose(leftWristPos, leftWristRot, rightWristPos, rightWristRot, headPose, headRot);
            current_time = Time.time;
        }

        return true;
    }

// Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        // Load SelectWorldScene
        local_ip = StartManipInterface.local_ip;
        cameraRig = GameObject.Find("OVRCameraRig").GetComponent<OVRCameraRig>();

        if (!TryGetComponents())
        {   
            current_text = "mising components";
            m_TimeText.text = current_text;
            enabled = false;
        }
        // set up socket
        try
        {   
            current_text = "Socket connecting...";
            m_TimeText.text = current_text;
            client = new UdpClient();
            client.Client.Bind(new IPEndPoint(IPAddress.Parse(local_ip), listen_port));
            //client.Client.Bind(new IPEndPoint(IPAddress.Any, 65432));
            remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
            current_text = "Socket connected";
            m_TimeText.text = current_text;
        }
        catch (Exception e)
        {   
            current_text = "Socket error: " + e.Message;
            m_TimeText.text = current_text;
        }

        // Create a folder with current time
        // folder_path = CoordinateFrame.folder_path;
        // Visualize coordinate frame pos
        // robot = GameObject.Find("panda_link0_vis");
        // robot_ee = GameObject.Find("panda_grasptarget_vis");
        // rleap = GameObject.Find("rpalm_vis");
        // GameObject frame = GameObject.Find("coordinate_vis");
        // frame.transform.position = CoordinateFrame.last_pos;
        // frame.transform.rotation = CoordinateFrame.last_rot;
        //  robot.GetComponent<ArticulationBody>().TeleportRoot(CoordinateFrame.last_pos, CoordinateFrame.last_rot);
        
        // Create sender socket
        sender = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        //targetEndPoint = new IPEndPoint(IPAddress.Parse(CoordinateFrame.remote_ip), sender_port);
        targetEndPoint = new IPEndPoint(IPAddress.Parse(StartManipInterface.pc_ip), sender_port);

    }
    

    private void SendHeadBimanualWristPose(Vector3 leftWristPos, Quaternion leftWristRot, Vector3 rightWristPos, Quaternion rightWristRot, Vector3 headPos, Quaternion headRot)
    {
        string message = "Bihand and Head: " + leftWristPos.x + "," + leftWristPos.y + "," + leftWristPos.z + "," + leftWristRot.x + "," + leftWristRot.y + "," + leftWristRot.z + "," + leftWristRot.w + ',';
        message = message + rightWristPos.x + "," + rightWristPos.y + "," + rightWristPos.z + "," + rightWristRot.x + "," + rightWristRot.y + "," + rightWristRot.z + "," + rightWristRot.w + ",";
        message = message + headPos.x + "," + headPos.y + "," + headPos.z + "," + headRot.x + "," + headRot.y + "," + headRot.z + "," + headRot.w;
        byte[] data = Encoding.UTF8.GetBytes(message);
        sender.SendTo(data, data.Length, SocketFlags.None, targetEndPoint);
    }
    // from ARCap
    // handedness: "L" or "R"
    // record: "Y" or "N"
    private void SendHeadWristPose( string handedness, string record, Vector3 wrist_pos, Quaternion wrist_rot, Vector3 head_pos, Quaternion head_orn)
    {
        string message = record + handedness+"Hand:" + wrist_pos.x + "," + wrist_pos.y + "," + wrist_pos.z + "," + wrist_rot.x + "," + wrist_rot.y + "," + wrist_rot.z + "," + wrist_rot.w;
        message = message + "," + head_pos.x + "," + head_pos.y + "," + head_pos.z + "," + head_orn.x + "," + head_orn.y + "," + head_orn.z + "," + head_orn.w;

        byte[] data = Encoding.UTF8.GetBytes(message);
        sender.SendTo(data, data.Length, SocketFlags.None, targetEndPoint);
    }   
    // Update is called once per frame
    protected void Update()
    {
        MainLoop();
    }

    protected void OnApplicationQuit()
    {
        if (client != null)
        {
            client.Close();
        }
    }
}
