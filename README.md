# Herramientas de visualización para redes neuronales

Páginas interactivas para el curso de Introducción a la IA. Son HTML y JavaScript estáticos, sin compilación, publicados con GitHub Pages.

| Carpeta | Página | Qué muestra |
|---|---|---|
| `cnn/` | CNN a la vista | Una CNN chica (2.346 pesos) entrenada con MNIST: filtros, mapas de activación, pooling, softmax y un lienzo para dibujar. Los pesos van dentro de la página. |
| `gpt/` | GPT a la vista | TinyStories-8M (GPT-Neo, 19,7 M de parámetros) corriendo en el navegador con código propio: tokenizador BPE de bytes, atención por bloque y cabeza, temperatura y generación. |

## Modelo de `gpt/`

- Pesos de [roneneldan/TinyStories-8M](https://huggingface.co/roneneldan/TinyStories-8M) en media precisión, partidos en `gpt/modelo/pesos-0.bin` y `pesos-1.bin` (38 MB en total). El orden de los tensores está en `gpt/modelo/meta.json`.
- Solo se guardan las primeras 256 posiciones; la página corta el texto en 120 tokens.
- Se verificó contra PyTorch: el tokenizador coincide en 155 textos y los logits y la atención coinciden en las frases de prueba.
- Referencia: Eldan y Li, «TinyStories: How Small Can Language Models Be and Still Speak Coherent English?» (2023).

## Ver localmente

```
python3 -m http.server
```

y abrir `http://localhost:8000/`.
