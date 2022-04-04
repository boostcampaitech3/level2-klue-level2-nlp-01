# KorEDA 참고
# https://github.com/catSirup/KorEDA/blob/master/eda.py
import random
import pickle
import re
import copy
from utils import *

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################
def random_deletion(words, subject_not, object_not, p):
    if len(words) == 1:
        return words

    new_words = []
    for idx, word in enumerate(words):
        r = random.uniform(0, 1)
        if (idx in subject_not) or (idx in object_not) or (r > p):
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################
def random_swap(words, subject_not, object_not, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words, subject_not, object_not)

    return new_words

def swap_word(new_words, subject_not, object_not):
    while True:
        random_idx_1 = random.randint(0, len(new_words)-1)
        if (random_idx_1 not in subject_not) and (random_idx_1 not in object_not):
            break
    random_idx_2 = random_idx_1
    counter = 0

    while random_idx_2 == random_idx_1:
        while True:
            random_idx_2 = random.randint(0, len(new_words)-1)
            if (random_idx_2 not in subject_not) and (random_idx_2 not in object_not):
                break
        counter += 1
        if counter > 3:
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

def EDA(sentence, subject_entity, object_entity, alpha_rs=0.1, p_rd=0.1, num_aug=5):
    subj_replace = '###SUBJ###'
    obj_replace = '###OBJ###'
    new_sentence = copy.deepcopy(sentence)
    new_sentence = new_sentence.replace(subject_entity, subj_replace)
    new_sentence = new_sentence.replace(object_entity, obj_replace)
    words = new_sentence.split(' ')
    words = [word for word in words if word != ""]
    subject_not_change_idx = []
    object_not_change_idx = []
    for idx, word in enumerate(words):
        if subj_replace in word:
            subject_not_change_idx.append(idx - 1)
            subject_not_change_idx.append(idx)
            subject_not_change_idx.append(idx + 1)
            break
    for idx, word in enumerate(words):
        if obj_replace in word:
            object_not_change_idx.append(idx - 1)
            object_not_change_idx.append(idx)
            object_not_change_idx.append(idx + 1)
            break
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug/2) + 1

    n_rs = max(1, int(alpha_rs*num_words))

    # rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, subject_not_change_idx, object_not_change_idx, n_rs)
        augmented_sentence = " ".join(a_words)
        augmented_sentence = augmented_sentence.replace(subj_replace, subject_entity)
        augmented_sentence = augmented_sentence.replace(obj_replace, object_entity)
        augmented_sentences.append(augmented_sentence)

    # rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, subject_not_change_idx, object_not_change_idx, p_rd)
        augmented_sentence = " ".join(a_words)
        augmented_sentence = augmented_sentence.replace(subj_replace, subject_entity)
        augmented_sentence = augmented_sentence.replace(obj_replace, object_entity)
        augmented_sentences.append(augmented_sentence)

    random.shuffle(augmented_sentences)
    augmented_sentences = augmented_sentences[:num_aug]

    return augmented_sentences

if __name__ == "__main__":
    train_data = load_data('./dataset/train/alternate_train.csv')
    sentences = train_data['sentence']
    subject_span = train_data['subject_span']
    object_span = train_data['object_span']
    subject_tag = train_data['subject_tag']
    object_tag = train_data['object_tag']
    labels = train_data['label']
    results = []
    for idx, sentence in enumerate(sentences):
        words = sentence.split(' ')
        words = [word for word in words if word != ""]
        if len(words) < 20 or labels[idx] == 'no_relation':
            continue
        subject_entity = sentence[subject_span[idx][0]:subject_span[idx][1]+1]
        object_entity = sentence[object_span[idx][0]:object_span[idx][1]+1]
        augmented_sentences = EDA(sentence, subject_entity, object_entity)
        results.append((augmented_sentences, subject_entity, subject_tag[idx], object_entity, object_tag[idx], labels[idx]))
    
    with open('augmented.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open('augmented.pkl', 'rb') as f:
        augmented_sentences = pickle.load(f)
    
    random.shuffle(augmented_sentences)

    with open('./dataset/train/augmented_train.csv', 'w') as f:
        for augmented_sentence_tuple in augmented_sentences:
            augmented_sentence, subject_entity, subject_tag, object_entity, object_tag, label = augmented_sentence_tuple
            for augmented in augmented_sentence:
                subj_start_idx = augmented.find(subject_entity)
                subj_end_idx = subj_start_idx + len(subject_entity) - 1
                obj_start_idx = augmented.find(object_entity)
                obj_end_idx = obj_start_idx + len(object_entity) - 1
                line_id = '-1'
                line_sentence = augmented
                line_subject = "\"{'word': '" + subject_entity + "', 'start_idx': " + str(subj_start_idx) + ", 'end_idx': " + str(subj_end_idx) + ", 'type': '" + subject_tag + "'}\""
                line_object = "\"{'word': '" + object_entity + "', 'start_idx': " + str(obj_start_idx) + ", 'end_idx': " + str(obj_end_idx) + ", 'type': '" + object_tag + "'}\""
                line_label = label
                line_source = 'Augmentation'
                line = line_id + '\t' + line_sentence + '\t' + line_subject + '\t' + line_object + '\t' + line_label + '\t' + line_source + '\n'
                f.write(line)
        print(subject_tag, object_tag)
    print('Finished!')