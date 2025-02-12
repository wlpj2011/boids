using UnityEngine;

[CreateAssetMenu]
public class BoidSettings : ScriptableObject
{
    public float minSpeed = 1f;
    public float maxSpeed = 4f;

    public float maxBound = 4f;

    public float turnFactor = 1f;
    public float visionRadius = 2f;
    public float avoidanceRadius = 1f;

    public float avoidanceStrength = 1f;
    public float cohesionStrength = 1f;
    public float alignmentStrength = 1f;

    public float boidSize = 0.2f;
}
