import morefun.composite_genotypes as cg
import morefun.grammars.upper_grammars as ugr
import morefun.randomness as rand

# autouse fixture
from morefun.tests.fixtures import remove_logger_sinks  # noqa


def get_grammar() -> ugr.Grammar:
    return ugr.Grammar(
        """
        start    : topology optimizer

        topology     : first_block middle_block~2
        first_block  : conv_block "fork"
        middle_block : "merge" conv_block "fork"

        conv_block : conv_layer batchnorm activation
                   | conv_layer batchnorm activation pooling

        conv_layer : "conv" "filter_count" ("32" | "64") "kernel_size" ("1" | "3" | "5" | "7") "stride" ("1" | "2")
        batchnorm  : "batchnorm"
        activation : relu | swish
        relu       : "relu"
        swish      : "swish"
        pooling    : maxpool | avgpool
        maxpool    : "maxpool" "pool_size" ("1" | "2") "stride" ("1" | "2")
        avgpool    : "avgpool" "pool_size" ("1" | "2") "stride" ("1" | "2")

        optimizer  : "adam" "learning_rate" ("0.001" | "0.003" | "0.005") "beta1" "0.9" "beta2" "0.999" "epsilon" "1e-07" "amsgrad" "false"
        """
    )


def test_uuids() -> None:
    """
    CompositeGenotypes are created with unique ids.
    """

    # this test looks silly, but was created because we found and fixed a bug
    # and wanted to ensure that it doesn't happen again

    grammar = get_grammar()
    rng = rand.create_rng()

    cg1 = cg.create_genotype(grammar, rng)
    cg2 = cg.create_genotype(grammar, rng)

    assert cg1.unique_id != cg2.unique_id
