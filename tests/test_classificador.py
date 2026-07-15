from classificador_sentimentos import ClassificadorNaiveBayes, avaliar, preprocessar
from dados_treino import DADOS_TESTE, DADOS_TREINO


def build_model():
    model = ClassificadorNaiveBayes()
    model.treinar(DADOS_TREINO)
    return model


def test_preprocess_marks_negation():
    assert "NAO_gostei" in preprocessar("Não gostei do atendimento")


def test_model_classifies_clear_examples():
    model = build_model()
    assert model.prever("produto excelente, adorei")[0] == "positivo"
    assert model.prever("experiência horrível, odiei")[0] == "negativo"


def test_evaluation_reaches_minimum_accuracy():
    assert avaliar(build_model(), DADOS_TESTE) >= 0.8
