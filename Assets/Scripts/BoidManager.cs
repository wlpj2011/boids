using UnityEngine;

public class BoidManager : MonoBehaviour
{
    public BoidSettings settings;
    Boid[] boids;


    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        boids = FindObjectsByType<Boid>(FindObjectsSortMode.None);
        foreach (Boid b in boids)
        {
            // Initialize each boid with the settings in BoidSettings
            b.Initialize(settings);
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (boids != null)
        {
            int numBoids = boids.Length;
            for (int i = 0; i < numBoids; i++)
            {
                // Update each boid with separation, alignment, and cohesion
                boids[i].SeparateFromBoids(boids, i);
                boids[i].AlignWithBoids(boids, i);
                boids[i].CohereWithBoids(boids, i);
                boids[i].UpdateBoid();
            }
        }
    }
}
