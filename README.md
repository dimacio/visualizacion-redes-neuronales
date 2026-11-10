# Herramientas de visualización para redes neuronales

Páginas interactivas para mirar por dentro redes neuronales chicas, con modelos de verdad que corren en el navegador. Son HTML y JavaScript estáticos, sin compilación, publicados con GitHub Pages.

| Carpeta | Página | Qué muestra |
|---|---|---|
| `audio/` | Audio a la vista | Convolución 1D sobre audio: ondas, kernels audibles, paso y pooling, espectrograma, y una CNN 1D (33.684 pesos) que reconoce clásica, metal, hip hop y jazz. Los pesos van dentro de la página. |
| `cnn/` | CNN a la vista | Una CNN chica (2.346 pesos) entrenada con MNIST: filtros, mapas de activación, pooling, softmax y un lienzo para dibujar. Los pesos van dentro de la página. |
| `gpt/` | GPT a la vista | TinyStories-8M (GPT-Neo, 19,7 M de parámetros) corriendo en el navegador con código propio: tokenizador BPE de bytes, atención por bloque y cabeza, temperatura y generación. |
| `llama/` | Demo Generativo | stories15M (arquitectura Llama, 15 M de parámetros) en el navegador: RoPE, RMSNorm y SwiGLU, atención por bloque y head, temperatura y generación. |

## Modelo de `audio/`

- CNN 1D sobre audio crudo a 16 kHz, ventanas de 3 s: cuatro capas convolucionales (16, 32, 32 y 64 canales; la primera con kernels de 64 muestras y paso 2), max pooling de 4, promedio en el tiempo y una capa lineal. La normalización por lotes está plegada en los pesos.
- Entrenada con los géneros clásica, metal, hip hop y jazz de GTZAN (Tzanetakis y Cook, 2002), con aumentos de ecualización, ruido y volumen al azar. Los clips 80 a 99 de cada género quedaron afuera para el test.
- Resultado: 90 % en esos 80 fragmentos de GTZAN y 72,5 % en 480 temas de [Free Music Archive](https://github.com/mdeff/fma) (120 por género), otra colección que la red nunca escuchó.
- Los 8 fragmentos de ejemplo son de Free Music Archive, con licencias Creative Commons que permiten obras derivadas; la página muestra autor, título y licencia de cada uno.
- La pasada hacia adelante corre en un worker del navegador. Se verificó contra PyTorch: mismos logits en la ventana de prueba.

## Modelo de `gpt/`

- Pesos de [roneneldan/TinyStories-8M](https://huggingface.co/roneneldan/TinyStories-8M) en media precisión, partidos en `gpt/modelo/pesos-0.bin` y `pesos-1.bin` (38 MB en total). El orden de los tensores está en `gpt/modelo/meta.json`.
- Solo se guardan las primeras 256 posiciones; la página corta el texto en 120 tokens.
- Se verificó contra PyTorch: el tokenizador coincide en 155 textos y los logits y la atención coinciden en las frases de prueba.
- Referencia: Eldan y Li, «TinyStories: How Small Can Language Models Be and Still Speak Coherent English?» (2023).

## Modelo de `llama/`

- Pesos de stories15M, de Andrej Karpathy ([llama2.c](https://github.com/karpathy/llama2.c), licencia MIT), entrenado con TinyStories. Arquitectura Llama: 6 bloques, 6 heads de 48 dimensiones, dimensión 288, MLP SwiGLU de 768, RMSNorm, RoPE y el vocabulario de 32.000 tokens de Llama 2.
- Reducidos a 8 bits con una escala por fila (15 MB), partidos en `llama/modelo/pesos.0.bin` a `pesos.2.bin`. La posición de cada tensor y el vocabulario están en `llama/modelo/modelo.json`.
- La pasada hacia adelante está escrita a mano dentro de la página, para mostrar cada número intermedio. La página corta el texto en 64 tokens.
- Se verificó contra `run.c` de llama2.c (mismo texto generado) y contra una implementación aparte en numpy (logits y atención con diferencias menores a 0,00001).

## Ver localmente

```
python3 -m http.server
```

y abrir `http://localhost:8000/`.
