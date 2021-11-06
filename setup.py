# Copyright 2021 Alexey Tochin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name='tf-seq2seq-losses',
    version="0.1.1",
    description='Tensorflow implementations for seq2seq Machine Learning model loss functions',
    long_description=(here / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/alexeytochin/tf-seq2seq-losses',
    author='Alexey Tochin',
    author_email='alexey.tochin@gmail.com',
    license='Apache 2.0',
    license_files=('LICENSE',),
    packages=['tf_seq2seq_losses'],
    install_requires=["tensorflow>=2.4.1", "numpy", "cached_property"],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='tensorflow, loss, ctc, connectionist temporal classification',
)