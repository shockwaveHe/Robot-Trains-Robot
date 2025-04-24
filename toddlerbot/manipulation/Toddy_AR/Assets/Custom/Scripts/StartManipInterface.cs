using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;
using Unity.XR.Oculus;
using TMPro;
using UnityEditor;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Net.NetworkInformation;
using System.Text;
using System.Timers;

public class StartManipInterface : MonoBehaviour
{   
    OVRCameraRig cameraRig;
    private TextMeshProUGUI init_text;
    public static string pc_ip;
    public static string local_ip;
    enum InitState
    {
        SETUP,
        CONFIRM
    }
    private  InitState myState = InitState.SETUP;
    private string default_text;
    private string current_text;
    private TouchScreenKeyboard overlayKeyboard;

    private static Timer timer;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        init_text = GameObject.Find("StartText").GetComponent<TextMeshProUGUI>();
        default_text = init_text.text;
        current_text = init_text.text;
        GetLocalIPAddress();
        overlayKeyboard = TouchScreenKeyboard.Open("", TouchScreenKeyboardType.Default);
        //overlayKeyboard.text = "Enter IP of your PC";
    }

    public void GetLocalIPAddress()
    {
        var host = Dns.GetHostEntry(Dns.GetHostName());
        foreach (var ip in host.AddressList)
        {
            if (ip.AddressFamily == AddressFamily.InterNetwork)
            {
                local_ip = ip.ToString();
                Debug.Log("local ip:" + local_ip);
            }
        }
    }
    private static void OnTimerElapsed(object sender, ElapsedEventArgs e)
    {
        Console.WriteLine("3 seconds passed!");
        timer.Dispose(); // Cleanup
    }
    // Update is called once per frame
    void Update()
    {
        //Debug.Log("Touch screen keyboard is null:" + (overlayKeyboard == null));
        //Debug.Log("Touch screen keyboard status:" + (overlayKeyboard.status));
        init_text.text = current_text;
        init_text.text +=  "\n" + "overlayKeyboard status: " + overlayKeyboard.status;
        init_text.text += "\n" + "overlayKeyboard active: " + overlayKeyboard.active;

        // For some reason the keyboard is not working properly on unity 6000
        //if (overlayKeyboard != null && overlayKeyboard.status == TouchScreenKeyboard.Status.Done)
        //{
        //    pc_ip = overlayKeyboard.text;
        //}
        if (overlayKeyboard != null && overlayKeyboard.active == false)
        {
            pc_ip = overlayKeyboard.text;
        }
        if (OVRInput.GetUp(OVRInput.RawButton.A))
        {
            // TODO
            if (myState == InitState.SETUP)
            {
                if (pc_ip == "")
                {
                    pc_ip = "10.128.156.68"; // default ip
                }
                //current_text = default_text;
                current_text = "IP setup! Confirming your IP . Press B to go back to previous step. Press A to continue...";
                current_text += "\n" + "pc_ip: " + pc_ip;
                current_text += "\n" + "local_ip: " + local_ip;
                current_text += "\n" + "overlayKeyboard status: " + overlayKeyboard.status;
                current_text += "\n" + "overlayKeyboard active: " + overlayKeyboard.active;
                init_text.text = current_text;
                myState = InitState.CONFIRM;
            }
            else if (myState == InitState.CONFIRM)
            {   
                if (pc_ip == "")
                {
                    pc_ip = "10.128.156.68"; // default ip
                }
                current_text = "Entering Toddy Scene...";
                current_text += "\n" + "pc_ip: " + pc_ip;
                current_text += "\n" + "local_ip: " + local_ip;
                timer = new Timer(3000);
                timer.Elapsed += OnTimerElapsed;
                timer.AutoReset = false;
                timer.Start();
                SceneManager.LoadScene("Toddy_scene");
            }
        }
        //if (OVRInput.GetUp(OVRInput.RawButton.A) && !init_enabled)
        //{
        //    string Not_setup_text = default_text + "\n" + "IP not setup!";
        //    Not_setup_text += "\n" + "pc_ip: " + pc_ip;
        //    Not_setup_text += "\n" + "local_ip: " + local_ip;
        //    Not_setup_text += "\n" + "overlayKeyboard status: " + overlayKeyboard.status;
        //    Not_setup_text += "\n" + "overlayKeyboard active: " + overlayKeyboard.active;
        //    init_text.text = Not_setup_text;
        //}
        if (OVRInput.GetUp(OVRInput.RawButton.B))
        {   
            if (myState == InitState.CONFIRM)
            {
                myState = InitState.SETUP;
                current_text = default_text;
                init_text.text = default_text;
                overlayKeyboard = TouchScreenKeyboard.Open("", TouchScreenKeyboardType.Default);
                //overlayKeyboard.text = "Enter IP of your PC";
            }
        }
    }
}
