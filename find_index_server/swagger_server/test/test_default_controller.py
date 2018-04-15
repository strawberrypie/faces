# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.embedding import Embedding  # noqa: E501
from swagger_server.test import BaseTestCase


class TestDefaultController(BaseTestCase):
    """DefaultController integration test stubs"""

    def test_get_sim_images(self):
        """Test case for get_sim_images

        Get sim images
        """
        vector = Embedding()
        response = self.client.open(
            '/v2/best_matches',
            method='POST',
            data=json.dumps(vector),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
