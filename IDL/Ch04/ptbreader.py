import collections

import tensorflow.compat.v1 as tf


def readWords(filename: str) -> list:
	tokens = ['<eos>']
	f = tf.io.gfile.GFile(filename, "r")
	tokens.extend(f.read().replace("\n", "<eos>").split()[:-1])
	return tokens


def buildVocab(filename='ptbdataset/ptb.train.txt') -> dict:
	data = readWords(filename)

	counter = collections.Counter(data)
	pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	words, _ = list(zip(*pairs))
	wordId = dict(zip(words, range(len(words))))
	return wordId


def fileToWordIds(filename: str, wordId: dict) -> list:
	data = readWords(filename)
	return [wordId[word] for word in data if word in wordId]


def loadWordIdsFromFiles():
	wordId = buildVocab()
	trainData = fileToWordIds('ptbdataset/ptb.train.txt', wordId)
	validData = fileToWordIds('ptbdataset/ptb.valid.txt', wordId)
	testData = fileToWordIds('ptbdataset/ptb.test.txt', wordId)
	return trainData, validData, testData, wordId
