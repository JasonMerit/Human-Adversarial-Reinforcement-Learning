using UnityEngine;
using System.Collections;

public class CameraShake : MonoBehaviour
{
    Vector3 originalPos;
    Coroutine shakeRoutine;
    public Vector2 durAndMag = new(0.1f, 0.2f);  // Hack

    void Awake()
    {
        originalPos = transform.localPosition;
    }

    public void Shake(Vector2 biasDirection = default, float duration = .2f, float magnitude = .2f)
    {
        if (shakeRoutine != null)
            StopCoroutine(shakeRoutine);

        if (biasDirection != default)
        {
            shakeRoutine = StartCoroutine(BiasShakeRoutine(duration, magnitude, biasDirection));
        }
        else
        {
            shakeRoutine = StartCoroutine(ShakeRoutine(duration, magnitude));
        }

    }

    IEnumerator BiasShakeRoutine(float duration, float magnitude, Vector2 biasDir)
    {
        float elapsed = 0f;

        while (elapsed < duration)
        {
            float t = elapsed / duration;
            float damper = 1f - t; // fade out

            // Strong directional push
            Vector2 directional = damper * magnitude * -biasDir;

            // Small random jitter
            Vector2 random = Random.insideUnitCircle * magnitude * 0.3f * damper;

            Vector3 offset = new Vector3(
                directional.x + random.x,
                directional.y + random.y,
                0f);

            transform.localPosition = originalPos + offset;

            elapsed += Time.deltaTime;
            yield return null;
        }

        transform.localPosition = originalPos;
    }
    
    IEnumerator ShakeRoutine(float duration, float magnitude)
    {
        float elapsed = 0f;

        while (elapsed < duration)
        {
            float x = Random.Range(-1f, 1f) * magnitude;
            float y = Random.Range(-1f, 1f) * magnitude;

            transform.localPosition = originalPos + new Vector3(x, y, 0);

            elapsed += Time.deltaTime;
            yield return null;
        }

        transform.localPosition = originalPos;
    }
}