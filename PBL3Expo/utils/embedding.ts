import AsyncStorage from '@react-native-async-storage/async-storage';

export type CatEmbedding = {
  id: string;
  name: string;
  embedding: number[];
};

const STORAGE_KEY = 'cat_embeddings';

export async function getAllEmbeddings(): Promise<CatEmbedding[]> {
  const json = await AsyncStorage.getItem(STORAGE_KEY);
  if (!json) return [];
  return JSON.parse(json);
}

export async function saveEmbedding(cat: CatEmbedding) {
  const all = await getAllEmbeddings();
  all.push(cat);
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(all));
}

export function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (normA * normB);
}

export async function findBestMatch(embedding: number[], threshold = 0.85): Promise<CatEmbedding | null> {
  const all = await getAllEmbeddings();
  let best: CatEmbedding | null = null;
  let bestScore = threshold;
  for (const cat of all) {
    const score = cosineSimilarity(embedding, cat.embedding);
    if (score > bestScore) {
      best = cat;
      bestScore = score;
    }
  }
  return best;
} 