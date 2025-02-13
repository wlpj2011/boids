using UnityEngine;

[CreateAssetMenu]
public class BoidSettings : ScriptableObject
{
    // speed for each boid
    public float minSpeed = 1f;
    public float maxSpeed = 4f;

    // Size of the box that the boids are contained in
    public float maxBound = 4f;

    // radii controlling boids
    public float turnFactor = 1f;
    public float visionRadius = 2f;
    public float avoidanceRadius = 1f;

    // strength of various boid forces
    public float avoidanceStrength = 1f;
    public float cohesionStrength = 1f;
    public float alignmentStrength = 1f;

    public float boidSize = 0.2f;
}
