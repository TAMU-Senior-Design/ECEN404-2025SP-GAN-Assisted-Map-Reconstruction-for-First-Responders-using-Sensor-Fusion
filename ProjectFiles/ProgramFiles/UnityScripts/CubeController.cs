using UnityEngine;
using Phidget22;
using Phidget22.Events;

public class CubeController : MonoBehaviour
{
    private Spatial spatial;
    private Quaternion targetRotation = Quaternion.identity;

    void Start()
    {
        // Initialize the Phidget Spatial sensor
        spatial = new Spatial();

        // Assign the AlgorithmData event to receive quaternion updates
        spatial.AlgorithmData += OnAlgorithmData;

        // Start the sensor
        spatial.Open(5000);

        // Set the data interval to the minimum for a more responsive experience
        spatial.DataInterval = spatial.MinDataInterval;
    }

    // Event handler to update the target quaternion
    private void OnAlgorithmData(object sender, SpatialAlgorithmDataEventArgs e)
    {
        targetRotation = new Quaternion(
            (float)e.Quaternion[0],
            (float)e.Quaternion[2],
            -(float)e.Quaternion[1],
            (float)e.Quaternion[3]
        );
    }

    void Update()
    {
        // Interpolate the cube's current rotation
        transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, Time.deltaTime * 20f);
    }

    void OnApplicationQuit()
    {
        // Close the sensor on application exit
        spatial.Close();
        spatial.Dispose();
    }
}
