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
        position += velocity * Time.deltaTime;
        facing = velocity.normalized;
        transform.localRotation = Quaternion.FromToRotation(transform.forward, -facing);
        transform.position = position;
    }

    public void SeparateBoid()
    {
        velocity += avoidance * settings.avoidanceStrength;
    }

    public void SeparateFromBoids(Boid[] boids, int index)
    {
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
        velocity += (alignment - velocity)* settings.alignmentStrength;
    }

    public void AlignWithBoids(Boid[] boids, int index)
    {
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
        velocity += (cohesion - position)* settings.cohesionStrength;
    }

    public void CohereWithBoids(Boid[] boids, int index)
    {
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
        float dist = 0f;
        dist += (transform.localPosition.x - other.transform.localPosition.x) * (transform.localPosition.x - other.transform.localPosition.x);
        dist += (transform.localPosition.y - other.transform.localPosition.y) * (transform.localPosition.y - other.transform.localPosition.y);
        dist += (transform.localPosition.z - other.transform.localPosition.z) * (transform.localPosition.z - other.transform.localPosition.z);
        return Sqrt(dist);
    }
    
    public float Speed()
    {
        return Sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
    }

    public void ClampSpeed()
    {
        float speed = Speed();
        float newSpeed = Clamp(speed, settings.minSpeed, settings.maxSpeed);
        velocity *= (newSpeed / speed);
    }

    public void TurnFromWall()
    {
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
