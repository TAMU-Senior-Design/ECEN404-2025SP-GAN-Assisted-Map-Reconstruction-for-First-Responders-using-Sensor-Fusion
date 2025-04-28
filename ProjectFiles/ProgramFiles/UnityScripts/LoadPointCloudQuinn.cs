using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using NativeWebSocket;
using System.Text;
using SimpleJSON;

public class LoadPointCloudQuinn : MonoBehaviour
{
    // Python server's IP address and port
    public string serverIPAddress = "192.168.1.1";
    public int serverPort = 5525;

    private WebSocket websocket;
    public Material pointMaterial;
    public Transform imuTransform;
    public Transform accumulationContainer;

    async void Start()
    {
        // Create an accumulation container if one hasn't been assigned
        if (accumulationContainer == null)
        {
            GameObject containerObj = new GameObject("AccumulationContainer");
            accumulationContainer = containerObj.transform;
        }

        // Initialize the WebSocket URL using the server IP and port
        string wsUrl = $"ws://{serverIPAddress}:{serverPort}";
        websocket = new WebSocket(wsUrl);

        websocket.OnOpen += () =>
        {
            Debug.Log("WebSocket connection opened.");
        };

        websocket.OnError += (errorMsg) =>
        {
            Debug.LogError("WebSocket error: " + errorMsg);
        };

        websocket.OnClose += (closeCode) =>
        {
            Debug.Log("WebSocket connection closed with code: " + closeCode);
        };

        websocket.OnMessage += (bytes) =>
        {
            string json = Encoding.UTF8.GetString(bytes);
            Debug.Log("Received JSON: " + json);
            ProcessPointCloudJSON(json);
        };

        await websocket.Connect();
        Debug.Log("WebSocket connect initiated to " + wsUrl);
    }

    void Update()
    {
        #if !UNITY_WEBGL || UNITY_EDITOR
                websocket.DispatchMessageQueue();
        #endif
    }

    void ProcessPointCloudJSON(string json)
    {
        JSONNode root = JSON.Parse(json);
        if (root == null)
        {
            Debug.LogWarning("Parsed JSON is null.");
            return;
        }

        JSONNode pointsNode = root;
        if (root.IsObject && root["points"] != null)
        {
            pointsNode = root["points"];
            Debug.Log("Found 'points' field in JSON. Count: " + pointsNode.Count);
        }

        if (!pointsNode.IsArray)
        {
            Debug.LogWarning("Expected an array of points in the JSON data, but got a different structure.");
            return;
        }

        List<Vector3> newVertices = new List<Vector3>();
        List<Color> newColors = new List<Color>();

        for (int i = 0; i < pointsNode.Count; i++)
        {
            JSONNode point = pointsNode[i];

            // Expecting only 3 values: x, y, z.
            if (point.Count < 3)
            {
                Debug.LogWarning("Point " + i + " does not have enough elements. Expected 3, got: " + point.Count);
                continue;
            }

            float x = point[0].AsFloat;
            float y = point[1].AsFloat;
            float z = point[2].AsFloat;

            newVertices.Add(new Vector3(y, z, -x));

            // Since no color data is provided by this sensor, assigned default white
            newColors.Add(Color.white);
        }

        if (newVertices.Count == 0)
        {
            Debug.LogWarning("No vertices found in received point cloud data.");
            return;
        }

        // Create and configure the mesh
        Mesh mesh = new Mesh();
        mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mesh.SetVertices(newVertices);
        mesh.SetIndices(Enumerable.Range(0, newVertices.Count).ToArray(), MeshTopology.Points, 0);
        mesh.SetColors(newColors);
        mesh.RecalculateBounds();

        // Create a new GameObject to display the point cloud
        GameObject scanObj = new GameObject("PointCloudScan");

        // Optionally orient the scan based on the current IMU transform (added 90 degree rotation)
        if (imuTransform != null)
        {
            scanObj.transform.rotation = imuTransform.rotation;
        }

        // Parent it under the fixed accumulation container
        scanObj.transform.SetParent(accumulationContainer, false);

        // Add mesh components
        MeshFilter mf = scanObj.AddComponent<MeshFilter>();
        MeshRenderer mr = scanObj.AddComponent<MeshRenderer>();
        mf.mesh = mesh;
        mr.material = pointMaterial;

        Debug.Log("Created new scan with " + newVertices.Count + " points, fixed in world space.");
    }

    async void OnApplicationQuit()
    {
        await websocket.Close();
    }
}
