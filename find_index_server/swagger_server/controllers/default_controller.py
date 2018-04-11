import connexion
import six

from swagger_server.models.embedding import Embedding  # noqa: E501
from swagger_server import util


def get_sim_images(vector):  # noqa: E501
    """Get sim images

     # noqa: E501

    :param vector: Embedding of image
    :type vector: dict | bytes

    :rtype: List[str]
    """
    if connexion.request.is_json:
        vector = Embedding.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'
