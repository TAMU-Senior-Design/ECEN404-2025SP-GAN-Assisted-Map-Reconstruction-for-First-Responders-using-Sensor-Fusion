using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using NativeWebSocket;
using System.Text;
using SimpleJSON;

public class LoadPointCloudRufus : MonoBehaviour
{
    private WebSocket websocket;
    public Material pointMaterial;
    public Transform imuTransform;
    public Transform accumulationContainer;

    async void Start()
    {
        // If no accumulation container is assigned, create one
        if (accumulationContainer == null)
        {
            GameObject containerObj = new GameObject("AccumulationContainer");
            accumulationContainer = containerObj.transform;
        }

        // Initialize WebSocket
        websocket = new WebSocket("ws://localhost:1212");

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
        Debug.Log("WebSocket connect initiated.");
    }

    void Update()
    {
        #if !UNITY_WEBGL || UNITY_EDITOR
                websocket.DispatchMessageQueue();
        #endif
    }

    // Parses JSON-formatted point cloud data (expects an array of arrays: [x, y, z, r, g, b]) and creates a new scan GameObject that is fixed in world space.

    void ProcessPointCloudJSON(string json)
    {
        Debug.Log("ProcessPointCloudJSON called with json: " + json);

        JSONNode root = null;
        try
        {
            root = JSON.Parse(json);
        }
        catch (System.Exception e)
        {
            Debug.LogError("JSON parsing exception: " + e.Message);
            return;
        }

        if (root == null)
        {
            Debug.LogWarning("Parsed JSON is null.");
            return;
        }

        // If the JSON is wrapped in an object with a points field, use that
        JSONNode pointsNode = root;
        if (root.IsObject && root["points"] != null)
        {
            pointsNode = root["points"];
            Debug.Log("Found 'points' field in JSON. Count: " + pointsNode.Count);
        }

        if (!pointsNode.IsArray)
        {
            Debug.LogWarning("Expected an array of points in the JSON data, but got something else.");
            return;
        }

        List<Vector3> newVertices = new List<Vector3>();
        List<Color> newColors = new List<Color>();

        for (int i = 0; i < pointsNode.Count; i++)
        {
            JSONNode point = pointsNode[i];
            if (point.Count < 6)
            {
                Debug.LogWarning("Point " + i + " does not have enough elements. Expected 6, got: " + point.Count);
                continue;
            }

            float x = point[0].AsFloat * 0.01f;
            float y = point[1].AsFloat * 0.01f;
            float z = point[2].AsFloat * 0.01f;
            float r = point[3].AsFloat / 255.0f;
            float g = point[4].AsFloat / 255.0f;
            float b = point[5].AsFloat / 255.0f;

            newVertices.Add(new Vector3(x, y, z));
            newColors.Add(new Color(r, g, b));

        }

        if (newVertices.Count == 0)
        {
            Debug.LogWarning("No vertices found in received point cloud data.");
            return;
        }

        // Create a new mesh for this scan
        Mesh mesh = new Mesh();
        mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mesh.SetVertices(newVertices);
        mesh.SetIndices(Enumerable.Range(0, newVertices.Count).ToArray(), MeshTopology.Points, 0);
        mesh.SetColors(newColors);
        mesh.RecalculateBounds();

        // Create a new GameObject to hold this scan
        GameObject scanObj = new GameObject("PointCloudScan");

        // Freeze the scan's orientation using the current IMU reading
        if (imuTransform != null)
        {
            scanObj.transform.rotation = imuTransform.rotation * Quaternion.Euler(0, -90, 0);
        }
        // Parent the scan under the fixed accumulation container
        scanObj.transform.SetParent(accumulationContainer, false);

        
        // Reposition the scan so that x = -4.5 and z = -3 (y remains unchanged), this is based on the camera's position in real life
        Vector3 newLocalPos = scanObj.transform.localPosition;
        newLocalPos.x = -4.5f;
        newLocalPos.z = -3f;
        scanObj.transform.localPosition = newLocalPos;

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
