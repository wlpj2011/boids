using UnityEngine;

public class Spawner : MonoBehaviour
{
    public Boid boidPrefab;
    public float spawnRadius = 10f;
    public int spawnCount = 10;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Awake()
    {
        for (int i = 0; i < spawnCount; i++)
        {
            // Spawn boids at random places within a sphere facing in a random direction
            Vector3 pos = transform.position + Random.insideUnitSphere * spawnRadius;
            Boid boid = Instantiate(boidPrefab);
            boid.transform.position = pos;
            boid.transform.forward = Random.insideUnitSphere;
        } 
    }
}
