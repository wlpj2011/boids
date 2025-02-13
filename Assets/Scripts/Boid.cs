using UnityEngine;
using UnityEngine.UIElements;
using static UnityEngine.Mathf;

public class Boid : MonoBehaviour
{
    BoidSettings settings;

    public Vector3 position;
    public Vector3 facing;
    Vector3 velocity;
    Vector3 acceleration;

    Vector3 avoidance;
    Vector3 alignment;
    Vector3 cohesion;

    public void Initialize(BoidSettings settings)
    {
        this.settings = settings;

        // Set the starting speed to be the average of the min and max and set the rest of the initial settings
        float startingSpeed = (settings.minSpeed + settings.maxSpeed) / 2;
        position = transform.position;
        facing = transform.right;
        velocity = transform.forward * startingSpeed;
        transform.localScale = Vector3.one * settings.boidSize;
    }

    public void UpdateBoid()
    {
        AlignBoid();
        SeparateBoid();
        CohereBoid();
        TurnFromWall();
        ClampSpeed();

        // Set the new facing direction and position
        position += velocity * Time.deltaTime;
        facing = velocity.normalized;
        transform.localRotation = Quaternion.FromToRotation(transform.forward, -facing);
        transform.position = position;
    }

    public void SeparateBoid()
    {
        // adjust velovity to separate from other boids
        velocity += avoidance * settings.avoidanceStrength;
    }

    public void SeparateFromBoids(Boid[] boids, int index)
    {
        // separate from other boids by aiming away from nearby boids
        avoidance = Vector3.zero;
        for (int i = 0; i < boids.Length; i++)
        {
            if (i != index)
            {
                Boid other = boids[i];
                if (DistanceToOther(other) < settings.avoidanceRadius)
                {
                    avoidance += position - other.position;
                }
            }
        }
    }

    public void AlignBoid()
    {
        // align with other boids in sight
        velocity += (alignment - velocity)* settings.alignmentStrength;
    }

    public void AlignWithBoids(Boid[] boids, int index)
    {
        // align with the average direction of nearby boids
        alignment = Vector3.zero;
        int neighboringBoids = 0;
        for (int i = 0; i < boids.Length; i++)
        {
            if (i != index)
            {
                Boid other = boids[i];
                if (DistanceToOther(other) < settings.visionRadius)
                {
                    alignment += velocity - other.velocity;
                    neighboringBoids += 1;
                }
            }
        }
        if (neighboringBoids != 0)
        {
            alignment /= (float)neighboringBoids;
        }
        else
        {
            alignment = Vector3.zero;
        }
    }

    public void CohereBoid()
    {
        // cohere with other boids in sight
        velocity += (cohesion - position)* settings.cohesionStrength;
    }

    public void CohereWithBoids(Boid[] boids, int index)
    {
        // cohere with the average location of nearby boids
        cohesion = Vector3.zero;
        int neighboringBoids = 0;
        for (int i = 0; i < boids.Length; i++)
        {
            if (i != index)
            {
                Boid other = boids[i];
                if (DistanceToOther(other) < settings.visionRadius)
                {
                    cohesion += position - other.position;
                    neighboringBoids += 1;
                }
            }
        }
        if (neighboringBoids != 0)
        {
            cohesion /= (float)neighboringBoids;
        }
        else
        {
            cohesion = Vector3.zero;
        }
    }
    
    public float DistanceToOther(Boid other)
    {
        // Calculate the distance to another boid
        float dist = 0f;
        dist += (transform.localPosition.x - other.transform.localPosition.x) * (transform.localPosition.x - other.transform.localPosition.x);
        dist += (transform.localPosition.y - other.transform.localPosition.y) * (transform.localPosition.y - other.transform.localPosition.y);
        dist += (transform.localPosition.z - other.transform.localPosition.z) * (transform.localPosition.z - other.transform.localPosition.z);
        return Sqrt(dist);
    }
    
    public float Speed()
    {
        // Calculate the current boid's strength
        return Sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
    }

    public void ClampSpeed()
    {
        // Clamp the velocity of the current boid between the min and max speed
        float speed = Speed();
        float newSpeed = Clamp(speed, settings.minSpeed, settings.maxSpeed);
        velocity *= (newSpeed / speed);
    }

    public void TurnFromWall()
    {
        // Turn back towards the center if outside the max bounds
        if (position.x < -settings.maxBound)
        {
            velocity.x += settings.turnFactor;
        }
        else if (position.x > settings.maxBound)
        {
            velocity.x -= settings.turnFactor;
        }
        if (position.y < -settings.maxBound)
        {
            velocity.y += settings.turnFactor;
        }
        else if (position.y > settings.maxBound)
        {
            velocity.y -= settings.turnFactor;
        }
        if (position.z < -settings.maxBound)
        {
            velocity.z += settings.turnFactor;
        }
        else if (position.z > settings.maxBound)
        {
            velocity.z -= settings.turnFactor;
        }
    }
}
